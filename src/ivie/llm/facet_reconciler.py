"""Reconcile independently generated world facets into one coherent GeneratedWorld.

Adapted from the ivie-msc implementation (Micaela Vaucher's master's thesis repository).

The four facet calls (locations, items, characters, objective) never see each other's output -
that is what lets them run concurrently. They agree on NAMES, because the skeleton fixes those
up front, but they can still disagree about facts only one of them should own. The most common
disagreement is item placement: the locations facet drops the Brass Key in the Hall while the
characters facet hands it to Mira.

Every such disagreement is settled here, deterministically, with no LLM involved - which is why
this module imports nothing but the data models and is covered by ordinary unit tests.

The headline rule is CHARACTER INVENTORY WINS: an NPC holding an item is usually load-bearing
for a puzzle or a trade, whereas an item lying in a room is more often incidental. Every
resolution is recorded in the returned report, so the disagreement rate between facets is
visible rather than silently patched over.
"""

from collections import deque
from typing import List, Optional

from .structured_data_models import (
    CharacterInteraction,
    GeneratedWorld,
    ItemReward,
    PassageReward,
    WorldLocationsFacet,
    WorldItemsFacet,
    WorldCharactersFacet,
    WorldObjectiveFacet,
)


class ReconciliationReport:
    """What the reconciler had to change, grouped by the kind of problem it fixed.

    `conflicts` counts genuine disagreements between two facets (the same item placed twice).
    `repairs` counts one facet referring to something that does not exist, or omitting
    something it should have covered. The distinction matters: a high conflict rate means the
    facet prompts overlap and should be tightened, while a high repair rate means a facet is
    ignoring its roster.
    """

    def __init__(self) -> None:
        self.conflicts: List[str] = []
        self.repairs: List[str] = []

    def conflict(self, message: str) -> None:
        self.conflicts.append(message)

    def repair(self, message: str) -> None:
        self.repairs.append(message)

    @property
    def total(self) -> int:
        return len(self.conflicts) + len(self.repairs)

    @property
    def messages(self) -> List[str]:
        return [f"conflict: {m}" for m in self.conflicts] + [f"repair: {m}" for m in self.repairs]

    def log(self, prefix: str = "[RECONCILE]") -> None:
        if self.total == 0:
            print(f"{prefix} facets agreed with no changes needed")
            return
        print(f"{prefix} {len(self.conflicts)} conflict(s), {len(self.repairs)} repair(s)")
        for message in self.messages:
            print(f"{prefix}   - {message}")


def _enum_value(value) -> str:
    return (value.value if hasattr(value, 'value') else str(value or '')).lower()


def _requirement_target(requirement) -> tuple:
    """The (kind, name) a blocked passage's requirement points at, or (None, None)."""
    if requirement is None:
        return None, None
    kind = _enum_value(getattr(requirement, 'requirement_type', None))
    for field, label in (('item_name', 'item'), ('puzzle_name', 'puzzle'),
                         ('character_name', 'character'), ('location_name', 'location')):
        name = getattr(requirement, field, None)
        if name:
            return (kind or label), name
    return kind or None, None


def _blocked_passage_is_enforceable(blocked, *, item_names, puzzle_names, character_names,
                                   location_names) -> tuple:
    """Whether the engine will actually enforce this block, and why not if it will not.

    world_builder only applies a block whose requirement resolves to a real Item or Puzzle - it
    drops the rest silently. A block the engine drops is worse than no block: the world's data
    says the passage is closed while play has it wide open.
    """
    kind, name = _requirement_target(getattr(blocked, 'required_to_unblock', None))
    if name is None:
        return False, "its requirement names nothing"
    if kind == 'item':
        if name not in item_names:
            return False, f"its required item '{name}' does not exist"
    elif kind == 'puzzle':
        if name not in puzzle_names:
            return False, f"its required puzzle '{name}' does not exist"
    elif kind == 'character':
        if name not in character_names:
            return False, f"its required character '{name}' does not exist"
        return False, f"a character requirement ('{name}') is not enforced by the engine"
    elif kind == 'location':
        if name not in location_names:
            return False, f"its required location '{name}' does not exist"
        return False, f"a location requirement ('{name}') is not enforced by the engine"
    else:
        return False, f"its requirement type '{kind}' is not one the engine enforces"
    return True, ""


def _distances_from(start: str, locations) -> dict:
    """Steps from `start` to every location, treating connections and blocked passages as
    two-way links (the same graph the world builder produces)."""
    names = {location.name for location in locations}
    neighbours = {name: set() for name in names}
    for location in locations:
        linked = list(location.connecting_locations) + [b.location for b in location.blocked_passages]
        for other in linked:
            if other in names and other != location.name:
                neighbours[location.name].add(other)
                neighbours[other].add(location.name)
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for other in neighbours.get(current, ()):
            if other not in distances:
                distances[other] = distances[current] + 1
                queue.append(other)
    return distances


def _farthest_location(start: str, locations) -> str:
    """The reachable location farthest from `start` (first in list order on ties)."""
    distances = _distances_from(start, locations)
    best, best_distance = start, 0
    for location in locations:
        distance = distances.get(location.name)
        if distance is not None and distance > best_distance:
            best, best_distance = location.name, distance
    return best


def _objective_components(objective, component_type: str) -> list:
    return [component.name for component in objective.components
            if _enum_value(component.component_type) == component_type]


def _avoid_trivial_objective(objective, player, characters, locations, report) -> None:
    """Make sure the objective is not already complete when the game starts.

    The facets are generated without seeing each other, so the characters facet can start the
    player at the objective's destination, or the locations facet can leave the target item at
    the starting location. check_objective runs after the first turn, so either would end the
    game immediately.
    """
    if len(locations) < 2:
        return
    objective_type = _enum_value(objective.type)
    locations_by_name = {location.name: location for location in locations}

    def move_item_away(item_name, away_from, reason):
        destination = _farthest_location(away_from, locations)
        if destination == away_from:
            return
        locations_by_name[destination].items.append(item_name)
        report.repair(f"objective item '{item_name}' {reason}; moved to '{destination}'")

    if objective_type == 'reach_location':
        for target in _objective_components(objective, 'location')[:1]:
            if player.location == target:
                new_start = _farthest_location(target, locations)
                if new_start != target:
                    report.repair(f"player started at the objective destination '{target}'; "
                                  f"start moved to '{new_start}'")
                    player.location = new_start

    elif objective_type == 'find_character':
        for target in _objective_components(objective, 'character')[:1]:
            character = next((c for c in characters if c.name == target), None)
            if character is not None and character.location == player.location:
                new_start = _farthest_location(character.location, locations)
                if new_start != character.location:
                    report.repair(f"player started next to the objective character '{target}'; "
                                  f"start moved to '{new_start}'")
                    player.location = new_start

    elif objective_type == 'get_item':
        for target in _objective_components(objective, 'item')[:1]:
            if target in player.inventory:
                player.inventory = [name for name in player.inventory if name != target]
                move_item_away(target, player.location, "started in the player's inventory")
            start = locations_by_name.get(player.location)
            if start is not None and target in start.items:
                start.items = [name for name in start.items if name != target]
                move_item_away(target, player.location, "was lying at the starting location")

    elif objective_type == 'deliver_an_item':
        items = _objective_components(objective, 'item')[:1]
        target_locations = _objective_components(objective, 'location')[:1]
        target_characters = _objective_components(objective, 'character')[:1]
        for item_name in items:
            for target in target_locations:
                destination = locations_by_name.get(target)
                if destination is not None and item_name in destination.items:
                    destination.items = [name for name in destination.items if name != item_name]
                    move_item_away(item_name, target, f"was already at its destination '{target}'")
            for target in target_characters:
                character = next((c for c in characters if c.name == target), None)
                if character is not None and item_name in character.inventory:
                    character.inventory = [name for name in character.inventory if name != item_name]
                    move_item_away(item_name, character.location, f"was already held by its recipient '{target}'")


def _force_gettable(items, required: dict, report) -> None:
    """Items the player must carry cannot be gettable=False, or Character.save_item refuses them."""
    for item in items:
        if item.name in required and not item.gettable:
            item.gettable = True
            report.repair(f"item '{item.name}' was not gettable but {required[item.name]}; made gettable")


def reconcile_world_facets(
    locations_facet: WorldLocationsFacet,
    items_facet: WorldItemsFacet,
    characters_facet: WorldCharactersFacet,
    objective_facet: WorldObjectiveFacet,
    *,
    puzzles: Optional[list] = None,
) -> tuple[GeneratedWorld, ReconciliationReport]:
    """Merge four facets into a GeneratedWorld, settling every disagreement deterministically.

    Returns the assembled world and a report of everything that had to be changed. The facets
    are not mutated - each is deep-copied first, so a caller may keep the raw facet output for
    comparison or logging.
    """
    report = ReconciliationReport()

    locations = [location.model_copy(deep=True) for location in locations_facet.locations]
    items = [item.model_copy(deep=True) for item in items_facet.items]
    characters = [character.model_copy(deep=True) for character in characters_facet.characters]
    player = characters_facet.player.model_copy(deep=True)
    objective = objective_facet.objective.model_copy(deep=True)

    location_names = {location.name for location in locations}
    item_names = {item.name for item in items}

    if not locations:
        # Nothing downstream can be repaired without at least one place to put it, and an
        # empty world is a generation failure, not something to paper over.
        raise ValueError("Locations facet produced no locations; cannot assemble a world")

    objective_items = set(_objective_components(objective, 'item'))

    # --- The player's starting location, resolved first ---------------------------------
    # Everything else falls back to it, so it has to be trustworthy before any other repair.
    if player.location not in location_names:
        fallback = locations[0].name
        report.repair(f"player start location '{player.location}' does not exist; moved to '{fallback}'")
        player.location = fallback
    player_start = player.location

    # --- Characters: valid locations, known inventory items, no duplicate holdings ------
    claimed_by_character: dict = {}
    for character in characters:
        if character.location not in location_names:
            report.repair(
                f"character '{character.name}' placed in non-existent location "
                f"'{character.location}'; moved to player start '{player_start}'"
            )
            character.location = player_start

        kept_inventory = []
        for item_name in character.inventory:
            if item_name not in item_names:
                report.repair(f"character '{character.name}' holds unknown item '{item_name}'; dropped")
                continue
            if item_name in claimed_by_character:
                report.conflict(
                    f"item '{item_name}' held by both '{claimed_by_character[item_name]}' and "
                    f"'{character.name}'; kept with '{claimed_by_character[item_name]}'"
                )
                continue
            claimed_by_character[item_name] = character.name
            kept_inventory.append(item_name)
        character.inventory = kept_inventory

    # The player's own starting inventory counts as a claim too, and outranks an NPC's:
    # an item the adventure hands the player up front must not also sit in a stranger's pocket.
    kept_player_inventory = []
    for item_name in player.inventory:
        if item_name not in item_names:
            report.repair(f"player holds unknown item '{item_name}'; dropped")
            continue
        holder = claimed_by_character.get(item_name)
        if holder is not None:
            report.conflict(
                f"item '{item_name}' held by both the player and '{holder}'; kept with the player"
            )
            for character in characters:
                if character.name == holder:
                    character.inventory = [n for n in character.inventory if n != item_name]
        claimed_by_character[item_name] = player.name
        kept_player_inventory.append(item_name)
    player.inventory = kept_player_inventory

    # --- Locations: known items, no duplicate placement, character inventory wins -------
    placed_in_location: dict = {}
    for location in locations:
        kept_items = []
        for item_name in location.items:
            if item_name not in item_names:
                report.repair(f"location '{location.name}' contains unknown item '{item_name}'; dropped")
                continue
            holder = claimed_by_character.get(item_name)
            if holder is not None:
                # The headline rule: an item a character carries is not also lying on the floor.
                report.conflict(
                    f"item '{item_name}' both lies in '{location.name}' and is held by "
                    f"'{holder}'; character inventory wins"
                )
                continue
            if item_name in placed_in_location:
                report.conflict(
                    f"item '{item_name}' placed in both '{placed_in_location[item_name]}' and "
                    f"'{location.name}'; kept in '{placed_in_location[item_name]}'"
                )
                continue
            placed_in_location[item_name] = location.name
            kept_items.append(item_name)
        location.items = kept_items

    # --- Orphan items: in the item list but nowhere in the world ------------------------
    # An unplaced item is unreachable, which can quietly make an objective impossible, so each
    # one is given a home. Ordinary items go to the player's starting location; an objective
    # item goes to the location farthest from it, or the objective would be trivial.
    locations_by_name = {location.name: location for location in locations}
    for item in items:
        if item.name in claimed_by_character or item.name in placed_in_location:
            continue
        if item.name in objective_items:
            home = _farthest_location(player_start, locations)
            where = "the location farthest from the player start"
        else:
            home = player_start
            where = "player start"
        locations_by_name[home].items.append(item.name)
        placed_in_location[item.name] = home
        report.repair(f"item '{item.name}' was placed nowhere; put in {where} '{home}'")

    # --- Topology: drop dangling links, then make every connection symmetric ------------
    character_names = {character.name for character in characters}
    for location in locations:
        kept_connections = []
        for connected_name in location.connecting_locations:
            if connected_name == location.name:
                report.repair(f"location '{location.name}' connects to itself; dropped")
                continue
            if connected_name not in location_names:
                report.repair(
                    f"location '{location.name}' connects to non-existent '{connected_name}'; dropped"
                )
                continue
            if connected_name in kept_connections:
                continue
            kept_connections.append(connected_name)
        location.connecting_locations = kept_connections

        kept_blocked = []
        for blocked in location.blocked_passages:
            if blocked.location not in location_names:
                report.repair(
                    f"location '{location.name}' blocks a passage to non-existent "
                    f"'{blocked.location}'; dropped"
                )
                continue
            # No puzzles exist yet at this stage, so a puzzle requirement here can never resolve.
            enforceable, why_not = _blocked_passage_is_enforceable(
                blocked, item_names=item_names, puzzle_names=set(),
                character_names=character_names, location_names=location_names,
            )
            if not enforceable:
                report.repair(
                    f"blocked passage '{location.name}' → '{blocked.location}' would not be "
                    f"enforced ({why_not}); dropped so the data matches what the engine builds"
                )
                continue
            kept_blocked.append(blocked)
        location.blocked_passages = kept_blocked

    # A blocked passage is still a passage: the engine needs the destination listed as a
    # connection, otherwise unblocking it leads nowhere.
    for location in locations:
        for blocked in location.blocked_passages:
            if blocked.location not in location.connecting_locations:
                location.connecting_locations.append(blocked.location)
                report.repair(
                    f"blocked passage '{location.name}' → '{blocked.location}' was missing from "
                    f"connections; added"
                )

    # Symmetry, applied deterministically instead of being asked for in the prompt and
    # checked afterwards - this removes a whole category of generation retry.
    for location in locations:
        for connected_name in list(location.connecting_locations):
            neighbour = locations_by_name[connected_name]
            if location.name not in neighbour.connecting_locations:
                neighbour.connecting_locations.append(location.name)
                report.repair(
                    f"connection '{location.name}' → '{connected_name}' was one-way; "
                    f"added the return link"
                )

    # --- A clue the player already holds could never be discovered -----------------------
    # Discovery fires when the player ACQUIRES a clue's item. An item sitting in the starting
    # inventory is never acquired, so its clue stays undiscovered forever and the mystery can
    # never complete. Moving it into the world also restores the point of a clue.
    clue_items = {clue.associated_item for clue in (getattr(objective, 'mystery_clues', None) or [])}
    for item_name in list(player.inventory):
        if item_name not in clue_items:
            continue
        player.inventory = [name for name in player.inventory if name != item_name]
        claimed_by_character.pop(item_name, None)
        locations_by_name[player_start].items.append(item_name)
        placed_in_location[item_name] = player_start
        report.repair(
            f"mystery clue item '{item_name}' started in the player's inventory, where its clue "
            f"could never be discovered; moved to '{player_start}' so it can be found"
        )

    # --- Mystery clues must agree with where items actually ended up ---------------------
    # The objective facet writes "clue item X is in location Y", but placement is settled above,
    # and character-inventory-wins routinely moves X into an NPC's hands. world_builder accepts a
    # clue only if the item is genuinely in the named location, so a stale item_location silently
    # invalidated every clue - and a mystery with no valid clues builds NO OBJECTIVE AT ALL.
    for clue in getattr(objective, 'mystery_clues', None) or []:
        if clue.associated_item not in item_names:
            report.repair(
                f"mystery clue '{clue.name}' names unknown item '{clue.associated_item}'; "
                f"it will be dropped when the world is built"
            )
            continue

        holder = claimed_by_character.get(clue.associated_item)
        actual_location = placed_in_location.get(clue.associated_item)
        if holder is not None:
            if clue.item_location:
                # Clearing it is what saves the clue: the builder accepts a clue with no
                # item_location, but rejects one naming a location the item is not in.
                report.repair(
                    f"mystery clue '{clue.name}' said '{clue.associated_item}' was in "
                    f"'{clue.item_location}', but {holder} holds it; cleared item_location"
                )
                clue.item_location = None
        elif actual_location is not None and clue.item_location != actual_location:
            report.repair(
                f"mystery clue '{clue.name}' said '{clue.associated_item}' was in "
                f"'{clue.item_location}', but it is in '{actual_location}'; corrected"
            )
            clue.item_location = actual_location

    # --- The objective must not be complete before the first move ------------------------
    _avoid_trivial_objective(objective, player, characters, locations, report)

    # --- Items the player has to carry must be gettable ----------------------------------
    must_be_gettable = {name: "it is part of the objective" for name in objective_items}
    for location in locations:
        for blocked in location.blocked_passages:
            kind, name = _requirement_target(blocked.required_to_unblock)
            if kind == 'item':
                must_be_gettable.setdefault(name, f"it unlocks the passage to '{blocked.location}'")
    for character in characters:
        interaction = character.interaction
        if interaction is None:
            continue
        if interaction.gives_item in item_names:
            must_be_gettable.setdefault(interaction.gives_item, f"'{character.name}' gives it to the player")
        for requirement in interaction.requires:
            kind, name = _requirement_target(requirement)
            if kind == 'item' and name in item_names:
                must_be_gettable.setdefault(name, f"'{character.name}' asks the player for it")
    _force_gettable(items, must_be_gettable, report)

    # The items facet guesses is_objective_target without seeing the objective; set it from the objective
    for item in items:
        item.is_objective_target = item.name in objective_items or item.name in clue_items

    world = GeneratedWorld(
        locations=locations,
        items=items,
        characters=characters,
        puzzles=list(puzzles) if puzzles else [],
        player=player,
        objective=objective,
        dependency_chains=[],
        world_theme=objective_facet.world_theme,
        narrative_context=objective_facet.narrative_context,
    )

    return world, report


def merge_puzzle_stage(world, stage_result, *, max_puzzles: Optional[int] = None) -> tuple:
    """Fold a PuzzleStageResult into a finished world, adding only what the stage may add.

    The locations, items, characters and objective are carried over untouched apart from the
    wiring the new puzzles need. That is the whole point: the stage this replaces took the world
    in and emitted it back out, which let it redistribute items, invent locations and break
    connectivity - and the pipeline could only detect that afterwards and retry, at the cost of
    a full world regeneration each time.

    Connectivity cannot regress here either, because a blocked passage only ever ADDS a link
    (the destination is appended to connecting_locations if it was missing).

    Returns the new world and a report of anything the stage got wrong.
    """
    report = ReconciliationReport()

    world = world.model_copy(deep=True)
    location_names = {location.name for location in world.locations}
    item_names = {item.name for item in world.items}
    character_names = {character.name for character in world.characters}
    locations_by_name = {location.name: location for location in world.locations}
    characters_by_name = {character.name: character for character in world.characters}

    # --- Puzzles: every element a puzzle names must already exist ------------------------
    kept_puzzles = []
    seen_puzzle_names = set()
    for puzzle in stage_result.puzzles:
        if puzzle.name in seen_puzzle_names:
            report.repair(f"puzzle '{puzzle.name}' was returned twice; kept the first")
            continue

        dangling = []
        if puzzle.location is not None and puzzle.location not in location_names:
            dangling.append(f"location '{puzzle.location}'")
        if puzzle.proposed_by_character is not None and puzzle.proposed_by_character not in character_names:
            dangling.append(f"character '{puzzle.proposed_by_character}'")
        if puzzle.proposed_by_location is not None and puzzle.proposed_by_location not in location_names:
            dangling.append(f"proposing location '{puzzle.proposed_by_location}'")

        if dangling:
            # A puzzle that lives somewhere that does not exist can never be found by the player
            report.repair(f"puzzle '{puzzle.name}' references non-existent {', '.join(dangling)}; dropped")
            continue

        puzzle = puzzle.model_copy(deep=True)
        kept_rewards = []
        for reward in puzzle.rewards:
            if isinstance(reward, PassageReward) and (
                    reward.from_location not in location_names or reward.to_location not in location_names):
                report.repair(
                    f"puzzle '{puzzle.name}' rewards a passage '{reward.from_location}' → "
                    f"'{reward.to_location}' between non-existent locations; reward dropped"
                )
                continue
            kept_rewards.append(reward)
        puzzle.rewards = kept_rewards

        seen_puzzle_names.add(puzzle.name)
        kept_puzzles.append(puzzle)

    if max_puzzles is not None and len(kept_puzzles) > max_puzzles:
        dropped = [puzzle.name for puzzle in kept_puzzles[max_puzzles:]]
        report.repair(f"stage returned {len(kept_puzzles)} puzzles but the world allows {max_puzzles}; "
                      f"dropped {dropped}")
        kept_puzzles = kept_puzzles[:max_puzzles]

    world.puzzles = kept_puzzles
    puzzle_names = {puzzle.name for puzzle in world.puzzles}

    # --- Blocked passage additions ------------------------------------------------------
    for addition in stage_result.blocked_passage_additions:
        origin = locations_by_name.get(addition.from_location)
        if origin is None:
            report.repair(
                f"blocked passage addition from non-existent location "
                f"'{addition.from_location}'; dropped"
            )
            continue
        if addition.passage.location not in location_names:
            report.repair(
                f"blocked passage '{addition.from_location}' → "
                f"'{addition.passage.location}' names a non-existent destination; dropped"
            )
            continue
        if addition.passage.location == addition.from_location:
            report.repair(f"blocked passage from '{addition.from_location}' to itself; dropped")
            continue
        if any(existing.location == addition.passage.location for existing in origin.blocked_passages):
            report.conflict(
                f"passage '{addition.from_location}' → '{addition.passage.location}' is already "
                f"blocked; kept the existing block"
            )
            continue

        enforceable, why_not = _blocked_passage_is_enforceable(
            addition.passage, item_names=item_names, puzzle_names=puzzle_names,
            character_names=character_names, location_names=location_names,
        )
        if not enforceable:
            report.repair(
                f"blocked passage addition '{addition.from_location}' → "
                f"'{addition.passage.location}' would not be enforced ({why_not}); dropped"
            )
            continue

        origin.blocked_passages.append(addition.passage.model_copy(deep=True))
        # A blocked passage is still a passage, so the destination has to be a connection or
        # unblocking it would lead nowhere.
        if addition.passage.location not in origin.connecting_locations:
            origin.connecting_locations.append(addition.passage.location)
        destination = locations_by_name[addition.passage.location]
        if origin.name not in destination.connecting_locations:
            destination.connecting_locations.append(origin.name)

    # --- Derive each character's proposes_puzzle from the puzzles themselves -------------
    # `puzzle.proposed_by_character` already carries this fact, so the stage never has to
    # restate it on the character - which is what used to force it to re-emit the whole cast.
    # The game triggers character puzzles from interaction.proposes_puzzle.
    puzzle_by_proposer = {}
    for puzzle in world.puzzles:
        if puzzle.proposed_by_character:
            puzzle_by_proposer.setdefault(puzzle.proposed_by_character, puzzle.name)

    for character in world.characters:
        proposed = puzzle_by_proposer.get(character.name)
        if proposed is None:
            continue
        if character.interaction is None:
            character.interaction = CharacterInteraction(
                relevance_to_objective=f"Proposes the puzzle '{proposed}'.",
                interaction_text=None,
            )
            report.repair(
                f"character '{character.name}' proposes '{proposed}' but had no interaction; "
                f"created one"
            )
        if character.interaction.proposes_puzzle != proposed:
            character.interaction.proposes_puzzle = proposed

    # --- A character's puzzle rewards an item that character holds ------------------------
    # Otherwise the player could pick the reward up without ever solving the puzzle. Rewards
    # that do not exist yet are created later by verify_puzzle_rewards_and_fix.
    for puzzle in world.puzzles:
        proposer = characters_by_name.get(puzzle.proposed_by_character or "")
        if proposer is None:
            continue
        for reward in puzzle.rewards:
            if not isinstance(reward, ItemReward) or reward.item_name not in item_names:
                continue
            item_name = reward.item_name
            if item_name in proposer.inventory or item_name in world.player.inventory:
                continue
            for location in world.locations:
                location.items = [name for name in location.items if name != item_name]
            for character in world.characters:
                character.inventory = [name for name in character.inventory if name != item_name]
            proposer.inventory.append(item_name)
            report.repair(f"reward '{item_name}' of '{puzzle.name}' was not held by its proposer "
                          f"'{proposer.name}'; moved to their inventory")

    # --- Keys and rewards must be gettable ------------------------------------------------
    must_be_gettable = {}
    for location in world.locations:
        for blocked in location.blocked_passages:
            kind, name = _requirement_target(blocked.required_to_unblock)
            if kind == 'item':
                must_be_gettable.setdefault(name, f"it unlocks the passage to '{blocked.location}'")
    for puzzle in world.puzzles:
        for reward in puzzle.rewards:
            if isinstance(reward, ItemReward):
                must_be_gettable.setdefault(reward.item_name, f"it is the reward of '{puzzle.name}'")
    _force_gettable(world.items, must_be_gettable, report)

    return world, report
