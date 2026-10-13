"""Whether a generated world can be completed, found by playing it forward.

Facets are generated without seeing each other, so a world can be locked in a circle: in one
playtest the only key to the lighthouse was carried by a spirit standing inside the lighthouse.
Checking each element on its own (the key exists, it is gettable, someone holds it) misses that.

progress() plays the world with the engine's rules until nothing new opens: which places the
player can reach, which items they can get and which puzzles they can solve. unmet_goals() says
what the objective needs that this leaves out of reach, and ensure_solvable() repairs the usual
cause, a blocked passage whose requirement can only be met behind it, by unblocking it.

Rules mirrored from core/turn_engine.py:
- An item lying in a reachable place can be taken if it is gettable.
- A character in a reachable place hands over what they carry unless it is the reward of their
  own unsolved puzzle, or one of their requirements is unmet (an item the player must give them,
  a puzzle solved, the player at a place). Requirements the engine cannot check do not stop it.
- A puzzle can be solved once it can be presented: its character is reachable, its item is at
  hand, or it blocks a passage from a reachable place. Solving it grants its rewards.
- A blocked passage opens with its item once the player can get it, or when its puzzle is solved.
"""

from dataclasses import dataclass, field

from .facet_reconciler import ReconciliationReport, _enum_value, _requirement_target
from .structured_data_models import ItemReward, PassageReward


@dataclass
class Stage:
    """What one pass of the walk added: a step in which the world opens up."""
    reached: list = field(default_factory=list)
    obtained: list = field(default_factory=list)
    solved: list = field(default_factory=list)
    opened: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.reached or self.obtained or self.solved or self.opened)


@dataclass
class Progress:
    reachable: set = field(default_factory=set)
    """locations the player can reach"""
    obtainable: set = field(default_factory=set)
    """items the player can get"""
    solved: set = field(default_factory=set)
    """puzzles the player can solve"""
    opened: set = field(default_factory=set)
    """(from, to) blocked passages the player can open"""
    stages: list = field(default_factory=list)
    """what each pass added, in order: the order in which the world opens up"""


def _met(requirement, progress) -> bool:
    kind, name = _requirement_target(requirement)
    if kind == 'item':
        return name in progress.obtainable
    if kind == 'puzzle':
        return name in progress.solved
    if kind == 'location':
        return name in progress.reachable
    return True  # talking, persuading: the engine does not check it


def progress(world, solved=()) -> Progress:
    """Everything the player can reach, get and solve, playing the world forward to a fixpoint.

    solved: puzzles already solved in a world being played. Their rewards are already in the world,
    so they are neither solved again nor listed in the stages.
    """
    locations = {location.name: location for location in world.locations}
    gettable = {item.name for item in world.items if item.gettable}
    state = Progress(reachable={world.player.location}, obtainable=set(world.player.inventory), solved=set(solved))
    known = Progress(solved=set(solved))
    gates = {}  # puzzle name -> locations whose blocked passage it gates
    for location in world.locations:
        for blocked in location.blocked_passages:
            kind, name = _requirement_target(blocked.required_to_unblock)
            if kind == 'puzzle':
                gates.setdefault(name, set()).add(location.name)

    changed = True
    while changed:
        for name in list(state.reachable):
            location = locations.get(name)
            if location is None:
                continue
            state.obtainable |= {item for item in location.items if item in gettable}
            blocked = {b.location: b for b in location.blocked_passages}
            for passage in blocked.values():
                if (name, passage.location) not in state.opened and _met(passage.required_to_unblock, state):
                    state.opened.add((name, passage.location))
            for neighbour in location.connecting_locations:
                if neighbour in locations and (neighbour not in blocked or (name, neighbour) in state.opened):
                    state.reachable.add(neighbour)

        for character in world.characters:
            if character.location not in state.reachable:
                continue
            withheld = {reward.item_name for puzzle in world.puzzles
                        if puzzle.proposed_by_character == character.name and puzzle.name not in state.solved
                        for reward in puzzle.rewards if isinstance(reward, ItemReward)}
            requires = character.interaction.requires if character.interaction else []
            if all(_met(requirement, state) for requirement in requires):
                state.obtainable |= {item for item in character.inventory if item in gettable and item not in withheld}

        for puzzle in world.puzzles:
            if puzzle.name in state.solved:
                continue
            proposer = next((c for c in world.characters if c.name == puzzle.proposed_by_character), None)
            presentable = ((proposer is not None and proposer.location in state.reachable)
                           or puzzle.proposed_by_location in state.obtainable
                           or any(puzzle.proposed_by_location in locations[name].items for name in state.reachable if name in locations)
                           or bool(gates.get(puzzle.name, set()) & state.reachable))
            if not presentable:
                continue
            state.solved.add(puzzle.name)
            for reward in puzzle.rewards:
                if isinstance(reward, ItemReward):
                    state.obtainable.add(reward.item_name)
                elif isinstance(reward, PassageReward):
                    state.opened.add((reward.from_location, reward.to_location))
                    if reward.from_location in state.reachable:
                        state.reachable.add(reward.to_location)

        stage = Stage(reached=sorted(state.reachable - known.reachable), obtained=sorted(state.obtainable - known.obtainable),
                      solved=sorted(state.solved - known.solved), opened=sorted(state.opened - known.opened))
        if stage:
            state.stages.append(stage)
        known = Progress(set(state.reachable), set(state.obtainable), set(state.solved), set(state.opened))
        changed = bool(stage)
    return state


def unmet_goals(world, state: Progress = None) -> list:
    """What the objective needs that the player can never reach, as readable reasons."""
    state = state or progress(world)
    objective = world.objective
    kind = _enum_value(objective.type)
    by_kind = {}
    for component in objective.components:
        by_kind.setdefault(_enum_value(component.component_type), []).append(component.name)
    character_at = {character.name: character.location for character in world.characters}
    problems = []

    if kind == 'solve_mystery':
        # A clue is discovered by examining its item, which only needs it at hand or in view
        held_at = {item: character.location for character in world.characters for item in character.inventory}
        lying_at = {item: location.name for location in world.locations for item in location.items}
        for clue in objective.mystery_clues or []:
            item = clue.associated_item
            if item not in state.obtainable and held_at.get(item) not in state.reachable and lying_at.get(item) not in state.reachable:
                problems.append(f"the clue item '{item}' can never be reached")
        return problems

    if kind in ('get_item', 'deliver_an_item'):
        problems += [f"the item '{name}' can never be obtained" for name in by_kind.get('item', []) if name not in state.obtainable]
    if kind in ('reach_location', 'deliver_an_item'):
        problems += [f"the place '{name}' can never be reached" for name in by_kind.get('location', []) if name not in state.reachable]
    if kind in ('find_character', 'deliver_an_item'):
        problems += [f"the character '{name}' can never be reached" for name in by_kind.get('character', [])
                     if character_at.get(name) not in state.reachable]
    return problems


def ensure_solvable(world) -> tuple:
    """Unblock passages the player could only open from behind until the objective is reachable.

    Returns (world, report, unmet goals). A blocked passage on the frontier of what the player can
    reach whose requirement is still unmet when nothing else opens is a lock with its key out of
    reach; opening it keeps the world playable at the cost of one obstacle. Goals still unmet
    afterwards have another cause and are returned for the caller to treat as an error.
    """
    report = ReconciliationReport()
    world = world.model_copy(deep=True)
    while True:
        state = progress(world)
        problems = unmet_goals(world, state)
        if not problems:
            return world, report, []
        stuck = [(location, blocked) for location in world.locations if location.name in state.reachable
                 for blocked in location.blocked_passages
                 if blocked.location not in state.reachable and (location.name, blocked.location) not in state.opened]
        if not stuck:
            return world, report, problems
        # One at a time: opening one passage may be enough, and every obstacle kept is play
        location, blocked = stuck[0]
        kind, name = _requirement_target(blocked.required_to_unblock)
        location.blocked_passages = [b for b in location.blocked_passages if b is not blocked]
        report.repair(f"passage '{location.name}' -> '{blocked.location}' unblocked: its {kind} '{name}' "
                      f"could only be reached behind it, so {problems[0]}")
