"""Apply a player's turn to the world and record what actually happened.

The reasoning model only interprets the player's action (a TurnIntent). This module decides, with
the world's rules, which of the attempted changes happen, applies them, and returns a TurnOutcome:
the list of facts the narration is written from. Because the narration no longer comes from the
same model call that proposes the changes, it cannot claim a hand-over, an open door or a solved
puzzle the world does not have.

Rules enforced here:
- An item is taken only if it lies at the player's location and can be carried.
- A character hands over an item only if they are at the player's location and the item is not
  the reward of their unsolved puzzle, and their verifiable requirements are met (a required item
  they now hold, a required puzzle solved, the player at a required location).
- An item can be given or dropped only if the player holds it.
- A blocked passage opens with the item that blocks it, if the player holds it and does not say
  they use something else; a passage blocked by a puzzle opens when the puzzle is solved.
- Every attempt that does not happen is recorded with its reason, including items the player
  uses or claims to carry that they do not have, and "I solved it" without an answer: the
  narration is written from these facts, so it must not have to guess.
- A puzzle is presented when the player addresses its character, examines its item or tries its
  passage, and is solved only if the player's own words contain an accepted answer.
- A mystery clue is discovered when its item is obtained or examined. Once every clue is known,
  accusing the culprit solves the mystery; before that, an accusation lacks the evidence.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .world import Item, MysteryObjective, Puzzle


@dataclass
class TurnEvent:
    kind: str
    ok: bool
    text_es: str
    text_en: str

    def text(self, language: str) -> str:
        return self.text_es if language == 'es' else self.text_en


@dataclass
class TurnOutcome:
    events: List[TurnEvent] = field(default_factory=list)
    arrived_at: Optional[object] = None
    first_visit: bool = False
    presented_puzzles: List[Puzzle] = field(default_factory=list)

    def add(self, kind: str, ok: bool, es: str, en: str) -> None:
        # One attempt can reach the engine twice ("I shoot the lock with my revolver" is a use and
        # an opening); the fact is told once
        event = TurnEvent(kind, ok, es, en)
        if event not in self.events:
            self.events.append(event)

    def kinds(self) -> List[str]:
        return [event.kind for event in self.events]

    def render(self, language: str) -> str:
        if not self.events:
            return "- (ningún cambio en el mundo)" if language == 'es' else "- (no change in the world)"
        mark = {True: "✓", False: "✗"}
        return "\n".join(f"- {mark[event.ok]} {event.text(language)}" for event in self.events)


#---- Helpers --------------------------------------------------------------------

def _holder_of(world, item):
    return next((c for c in world.characters.values() if c is not world.player and item in c.inventory), None)


def _requirement_met(world, character, requirement) -> Optional[bool]:
    """True/False for requirements the engine can check; None for the ones it cannot (talking, persuading)."""
    kind = str(getattr(getattr(requirement, 'requirement_type', None), 'value', getattr(requirement, 'requirement_type', ''))).lower()
    if kind == 'item':
        item = world._find_object_flexible(getattr(requirement, 'item_name', '') or '')
        return item is not None and item in character.inventory
    if kind == 'puzzle':
        puzzle = world.puzzles.get(getattr(requirement, 'puzzle_name', '') or '')
        return puzzle is not None and world.puzzle_states.get(puzzle.name) == 'solved'
    if kind == 'location':
        return world.player.location.name == getattr(requirement, 'location_name', None)
    return None


def _puzzles_of(world, character) -> List[Puzzle]:
    proposed = getattr(getattr(character, 'interaction', None), 'proposes_puzzle', None)
    return [p for p in world.puzzles.values()
            if p.proposed_by_character == character.name or p.name == proposed]


def _handover_lock(world, character, item):
    """(reason_es, reason_en, puzzle_to_present) when the character may not hand the item over yet."""
    for puzzle in _puzzles_of(world, character):
        if world.puzzle_states.get(puzzle.name) == 'solved':
            continue
        rewards = [getattr(r, 'item_name', None) for r in puzzle.rewards]
        if item.name in rewards:
            return (f"{character.name} entrega {item.name} solo cuando se resuelve su acertijo '{puzzle.name}'",
                    f"{character.name} hands over {item.name} only once their puzzle '{puzzle.name}' is solved",
                    puzzle)
    interaction = getattr(character, 'interaction', None)
    for requirement in (getattr(interaction, 'requires', None) or []):
        if _requirement_met(world, character, requirement) is False:
            description = getattr(requirement, 'description', '')
            return (f"{character.name} todavía no lo entrega: antes pide {description}",
                    f"{character.name} does not hand it over yet: first they require {description}",
                    None)
    return None


def _puzzle_in_scene(world, puzzle) -> bool:
    location = world.player.location
    if puzzle.proposed_by_character:
        character = world.characters.get(puzzle.proposed_by_character)
        if character is not None and character.location is location:
            return True
    if puzzle.proposed_by_location:
        item = world._find_object_flexible(puzzle.proposed_by_location)
        if item is not None and (item in location.items or item in world.player.inventory):
            return True
    return any(obstacle is puzzle for (_, obstacle, _) in location.blocked_locations.values())


def _find_puzzle(world, name) -> Optional[Puzzle]:
    if not name:
        return None
    if name in world.puzzles:
        return world.puzzles[name]
    from ..llm.name_matching import normalize_name
    wanted = normalize_name(name)
    return next((p for p in world.puzzles.values() if normalize_name(p.name) == wanted), None)


#---- The turn -------------------------------------------------------------------

class _Turn:
    def __init__(self, world, intent, player_input):
        self.world = world
        self.intent = intent
        self.player_input = player_input or ""
        self.outcome = TurnOutcome()

    @property
    def player(self):
        return self.world.player

    def present(self, puzzle) -> None:
        if self.world.puzzle_states.get(puzzle.name) == 'solved' or puzzle in self.outcome.presented_puzzles:
            return
        self.world.puzzle_states[puzzle.name] = 'proposed'
        self.outcome.presented_puzzles.append(puzzle)
        self.world.update_hints_for_puzzle_activity(puzzle.name)
        self.outcome.add('puzzle_presented', True,
                         f"Se presenta el acertijo '{puzzle.name}': \"{puzzle.problem}\"",
                         f"The puzzle '{puzzle.name}' is presented: \"{puzzle.problem}\"")

    def discover_clue(self, item) -> None:
        objective = self.world.objective
        if not objective or not isinstance(objective[1], MysteryObjective):
            return
        clue = objective[1].discover_clue_for_item(item.name)
        if clue:
            found, total = objective[1].get_completion_progress()
            self.outcome.add('clue_discovered', True,
                             f"Pista del misterio descubierta ({found}/{total}) - {clue.name}: {clue.description}",
                             f"Mystery clue discovered ({found}/{total}) - {clue.name}: {clue.description}")

    # -- items ----------------------------------------------------------------------

    def move_item(self, moved) -> None:
        world, player = self.world, self.player
        item = world._find_object_flexible(moved.object_name)
        if item is None:
            self.outcome.add('item_missing', False, f"No existe ningún objeto llamado '{moved.object_name}'",
                             f"There is no item called '{moved.object_name}'")
            return
        recipient = world._find_character_flexible(moved.new_location)
        if recipient is None and world._is_player_destination(moved.new_location):
            self.obtain(item)
        elif recipient is not None:
            self.give(item, recipient)
        else:
            self.drop(item)

    def obtain(self, item) -> None:
        world, player, location = self.world, self.player, self.player.location
        if item in player.inventory:
            return
        holder = _holder_of(world, item)
        if holder is not None:
            if holder.location is not location:
                self.outcome.add('handover_refused', False, f"{holder.name}, que tiene {item.name}, no está en este lugar",
                                 f"{holder.name}, who has {item.name}, is not here")
                return
            lock = _handover_lock(world, holder, item)
            if lock:
                self.outcome.add('handover_refused', False, lock[0], lock[1])
                if lock[2] is not None:
                    self.present(lock[2])
                return
            if not item.gettable:
                self.outcome.add('handover_refused', False, f"{item.name} no se puede llevar", f"{item.name} cannot be carried")
                return
            player.save_item(item, holder)
            self.outcome.add('item_received', True, f"{holder.name} le entrega {item.name} al jugador",
                             f"{holder.name} hands {item.name} to the player")
            self.discover_clue(item)
            return
        if item in location.items:
            if not item.gettable:
                self.outcome.add('take_refused', False, f"{item.name} no se puede llevar: está fijo o es demasiado grande",
                                 f"{item.name} cannot be carried: it is fixed in place or too large")
                return
            player.save_item(item, location)
            self.outcome.add('item_taken', True, f"El jugador toma {item.name}", f"The player takes {item.name}")
            self.discover_clue(item)
            return
        self.outcome.add('take_refused', False, f"{item.name} no está al alcance del jugador en este lugar",
                         f"{item.name} is not within the player's reach here")

    def give(self, item, character) -> None:
        if item not in self.player.inventory:
            self.outcome.add('give_refused', False, f"El jugador no tiene {item.name}", f"The player does not have {item.name}")
            return
        if character.location is not self.player.location:
            self.outcome.add('give_refused', False, f"{character.name} no está en este lugar", f"{character.name} is not here")
            return
        self.player.give_item(character, item)
        self.outcome.add('item_given', True, f"El jugador le da {item.name} a {character.name}",
                         f"The player gives {item.name} to {character.name}")

    def drop(self, item) -> None:
        if item not in self.player.inventory:
            self.outcome.add('drop_refused', False, f"El jugador no tiene {item.name}", f"The player does not have {item.name}")
            return
        self.player.drop_item(item)
        self.outcome.add('item_dropped', True, f"El jugador deja {item.name} en {self.player.location.name}",
                         f"The player leaves {item.name} in {self.player.location.name}")

    # -- passages and movement ---------------------------------------------------------

    def open_passage(self, attempt) -> None:
        world, location = self.world, self.player.location
        destination = world._find_location_flexible(attempt.destination)
        if destination is None or destination.name not in location.blocked_locations:
            return  # not a blocked passage from here: nothing to open
        _, obstacle, _ = location.blocked_locations[destination.name]
        if isinstance(obstacle, Puzzle):
            self.outcome.add('passage_closed', False,
                             f"El paso hacia {destination.name} solo se abre resolviendo el acertijo '{obstacle.name}'",
                             f"The way to {destination.name} only opens by solving the puzzle '{obstacle.name}'")
            self.present(obstacle)
            return
        if attempt.using_item:
            tool = world._find_object_flexible(attempt.using_item)
            if tool is None or tool not in self.player.inventory:
                self.unavailable(tool, attempt.using_item)
                return
            if tool is not obstacle:
                self.outcome.add('passage_closed', False,
                                 f"El paso hacia {destination.name} no se abre con {tool.name}",
                                 f"The way to {destination.name} does not open with {tool.name}")
                return
        if obstacle in self.player.inventory:
            location.unblock_passage(destination)
            self.outcome.add('passage_opened', True, f"El jugador abre el paso hacia {destination.name} usando {obstacle.name}",
                             f"The player opens the way to {destination.name} using {obstacle.name}")
            return
        self.outcome.add('passage_closed', False,
                         f"El paso hacia {destination.name} sigue bloqueado: el jugador no tiene lo que lo abre",
                         f"The way to {destination.name} is still blocked: the player does not have what opens it")

    def go_to(self, location_name) -> None:
        world, player = self.world, self.player
        target = world._find_location_flexible(location_name)
        if target is None:
            self.outcome.add('move_refused', False, f"No hay ningún lugar llamado '{location_name}'",
                             f"There is no place called '{location_name}'")
            return
        if target is player.location:
            return
        if target in player.location.connecting_locations:
            self.outcome.first_visit = not target.visited
            player.move(target)
            self.outcome.arrived_at = target
            world.update_hints()
            self.outcome.add('moved', True, f"El jugador va a {target.name}", f"The player goes to {target.name}")
            return
        if target.name in player.location.blocked_locations:
            self.outcome.add('move_refused', False, f"El paso hacia {target.name} está bloqueado",
                             f"The way to {target.name} is blocked")
            _, obstacle, _ = player.location.blocked_locations[target.name]
            if isinstance(obstacle, Puzzle):
                self.present(obstacle)
            return
        self.outcome.add('move_refused', False, f"{target.name} no está conectado con {player.location.name}",
                         f"{target.name} is not connected to {player.location.name}")

    # -- characters, items examined and puzzles -----------------------------------------

    def address(self, character_name) -> None:
        character = self.world._find_character_flexible(character_name)
        if character is None:
            return
        if character.location is not self.player.location:
            self.outcome.add('character_absent', False, f"{character.name} no está en este lugar",
                             f"{character.name} is not here")
            return
        for puzzle in _puzzles_of(self.world, character):
            self.present(puzzle)

    def unavailable(self, item, name) -> None:
        """The player uses or mentions an item they do not have and cannot reach."""
        if item is not None and item in self.player.location.items:
            self.outcome.add('item_not_carried', False,
                             f"El jugador no lleva {item.name}: está en este lugar, pero primero tiene que tomarlo",
                             f"The player is not carrying {item.name}: it is here, but they have to take it first")
        elif item is None:
            self.outcome.add('item_missing', False,
                             f"El jugador no tiene '{name}': no está en su inventario ni en este lugar",
                             f"The player has no '{name}': it is neither in their inventory nor here")
        else:
            self.outcome.add('item_missing', False,
                             f"{item.name} no está al alcance: el jugador no lo tiene y no está en este lugar",
                             f"{item.name} is out of reach: the player does not have it and it is not here")

    def use(self, item_name) -> None:
        """Using an item needs it at hand: in the inventory or lying here. Otherwise the player does
        not have it, even if they say so ("I shoot with my revolver")."""
        item = self.world._find_object_flexible(item_name)
        if item is None or (item not in self.player.inventory and item not in self.player.location.items):
            self.unavailable(item, item_name)
            return
        self.examine(item.name)

    def examine(self, item_name) -> None:
        world, location = self.world, self.player.location
        item = world._find_object_flexible(item_name)
        if item is None:
            return  # scenery the world does not model: nothing to check
        holder = _holder_of(world, item)
        reachable = item in location.items or item in self.player.inventory or (holder is not None and holder.location is location)
        if not reachable:
            self.unavailable(item, item_name)
            return
        self.discover_clue(item)
        puzzle = world.find_puzzle_proposed_by_location(item.name)
        if puzzle is not None:
            self.present(puzzle)

    def answer_puzzles(self) -> None:
        """A puzzle is solved when the player's own words contain its answer. The interpreter's
        copy of the answer only tells a wrong answer apart from no answer at all."""
        world = self.world
        given = {}
        for attempt in self.intent.puzzle_attempts:
            puzzle = _find_puzzle(world, attempt.puzzle_name)
            if puzzle is not None:
                given[puzzle] = (attempt.answer or "").strip()
        in_scene = [p for p in world.puzzles.values()
                    if world.puzzle_states.get(p.name) == 'proposed' and _puzzle_in_scene(world, p)]
        for puzzle in list(dict.fromkeys(list(given) + in_scene)):
            if world.puzzle_states.get(puzzle.name) == 'solved':
                continue
            if puzzle.is_answered_by(self.player_input):
                self.solve(puzzle)
            elif puzzle in given and not given[puzzle]:
                self.outcome.add('puzzle_unanswered', False,
                                 f"El jugador dice haber resuelto el acertijo '{puzzle.name}' pero no da ninguna respuesta: sigue sin resolver",
                                 f"The player says they solved the puzzle '{puzzle.name}' but gives no answer: it remains unsolved")
            elif puzzle in given:
                self.outcome.add('puzzle_failed', False,
                                 f"La respuesta \"{given[puzzle]}\" al acertijo '{puzzle.name}' no es correcta",
                                 f"The answer \"{given[puzzle]}\" to the puzzle '{puzzle.name}' is not correct")

    def accuse(self, name) -> None:
        objective = self.world.objective
        mystery = objective[1] if objective and isinstance(objective[1], MysteryObjective) else None
        if mystery is None or not getattr(mystery, 'culprit', None) or mystery.is_completed():
            return
        character = self.world._find_character_flexible(name)
        accused = character.name if character is not None else name
        if not mystery.ready_to_accuse():
            found, total = mystery.get_completion_progress()
            self.outcome.add('accusation_premature', False,
                             f"El jugador acusa a {accused}, pero todavía no reunió las pruebas para sostener una acusación ({found}/{total} pistas)",
                             f"The player accuses {accused}, but has not yet gathered the evidence to back an accusation ({found}/{total} clues)")
        elif mystery.accuse(accused):
            self.outcome.add('mystery_solved', True,
                             f"El jugador acusa a {accused}, y las pruebas lo confirman: es el responsable. Lo que pasó: {mystery.mystery_solution}",
                             f"The player accuses {accused}, and the evidence confirms it: they are responsible. What happened: {mystery.mystery_solution}")
        else:
            self.outcome.add('accusation_failed', False,
                             f"El jugador acusa a {accused}, pero las pruebas no lo señalan: la acusación no se sostiene",
                             f"The player accuses {accused}, but the evidence does not point to them: the accusation does not hold")

    def solve(self, puzzle) -> None:
        world, player = self.world, self.player
        before = list(player.inventory)
        world.puzzle_states[puzzle.name] = 'solved'
        world._apply_puzzle_rewards(puzzle)
        world._unblock_passages_for_solved_puzzle(puzzle.name, puzzle)
        world.update_hints()
        self.outcome.add('puzzle_solved', True, f"El jugador resuelve el acertijo '{puzzle.name}'",
                         f"The player solves the puzzle '{puzzle.name}'")
        for item in player.inventory:
            if item not in before:
                self.outcome.add('item_received', True, f"Como recompensa, el jugador recibe {item.name}",
                                 f"As a reward, the player receives {item.name}")
                self.discover_clue(item)


def apply_turn(world, intent, player_input: str) -> TurnOutcome:
    """Apply what the player attempted, in the order a player would expect: handle items, open
    passages, move, then interactions and answers at the resulting location."""
    turn = _Turn(world, intent, player_input)
    for moved in intent.moved_objects:
        turn.move_item(moved)
    for attempt in intent.passages_to_open:
        turn.open_passage(attempt)
    if intent.location_changed and intent.location_changed.new_location:
        turn.go_to(intent.location_changed.new_location)
    for name in intent.characters_addressed:
        turn.address(name)
    for name in intent.examined_items:
        turn.examine(name)
    for name in intent.items_used:
        turn.use(name)
    turn.answer_puzzles()
    if intent.accusation:
        turn.accuse(intent.accusation)
    return turn.outcome
