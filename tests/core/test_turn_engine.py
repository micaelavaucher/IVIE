"""turn_engine.apply_turn: the engine decides what a player's attempts actually do."""

from src.ivie.core.turn_engine import apply_turn
from src.ivie.core.world import Character, Item, Location, MysteryClue, MysteryObjective, Puzzle, World
from src.ivie.llm.structured_data_models import (
    CharacterInteraction, ItemRequirement, ItemReward, MovedObject, PuzzleRequirement, TurnIntent,
)


def _world():
    """Hall (start) - Torre; Hall -> Despacho blocked by the Llave, which Marlowe holds."""
    hall, torre, despacho = Location("Hall", ["d"]), Location("Torre", ["d"]), Location("Despacho", ["d"])
    for a, b in ((hall, torre), (hall, despacho)):
        a.connecting_locations.append(b)
        b.connecting_locations.append(a)
    llave, diario, gema = Item("Llave del Despacho", ["d"]), Item("Diario", ["d"]), Item("Gema", ["d"])
    estatua, orbe = Item("Estatua", ["d"], gettable=False), Item("Orbe", ["d"])
    hall.items += [gema, estatua]
    torre.items.append(orbe)
    hall.block_passage(despacho, llave)
    player = Character("Jugador", ["d"], location=hall)
    marlowe = Character("Marlowe", ["d"], location=hall, inventory=[diario, llave])
    sabio = Character("Sabio", ["d"], location=torre, inventory=[])
    world = World(player)
    world.add_locations([hall, torre, despacho])
    world.add_items([llave, diario, gema, estatua, orbe])
    world.add_character(marlowe)
    world.add_character(sabio)
    return world


def _intent(**fields):
    fields.setdefault("action_summary", "s")
    moves = fields.pop("moves", [])
    fields["moved_objects"] = [MovedObject(object_name=item, new_location=to) for item, to in moves]
    if "go" in fields:
        fields["location_changed"] = {"new_location": fields.pop("go")}
    return TurnIntent(**fields)


def _names(items):
    return [item.name for item in items]


#---- Items ----------------------------------------------------------------------

def test_taking_a_loose_item():
    world = _world()
    outcome = apply_turn(world, _intent(moves=[("Gema", "Jugador")]), "tomo la gema")

    assert _names(world.player.inventory) == ["Gema"] and outcome.kinds() == ["item_taken"]


def test_a_fixed_item_or_an_item_elsewhere_cannot_be_taken():
    world = _world()
    outcome = apply_turn(world, _intent(moves=[("Estatua", "Jugador"), ("Orbe", "Jugador")]), "tomo todo")

    assert world.player.inventory == [] and outcome.kinds() == ["take_refused", "take_refused"]


def test_a_character_hands_over_an_item_with_no_condition():
    world = _world()
    outcome = apply_turn(world, _intent(moves=[("Llave del Despacho", "Jugador")]), "le pido la llave a Marlowe")

    assert "Llave del Despacho" in _names(world.player.inventory) and outcome.kinds() == ["item_received"]


def test_a_character_elsewhere_hands_over_nothing():
    world = _world()
    world.characters["Marlowe"].location = world.locations["Torre"]
    outcome = apply_turn(world, _intent(moves=[("Diario", "Jugador")]), "le pido el diario a Marlowe")

    assert world.player.inventory == [] and outcome.kinds() == ["handover_refused"]


def test_the_reward_of_an_unsolved_puzzle_is_withheld_and_the_puzzle_presented():
    world = _world()
    world.add_puzzle(Puzzle("Acertijo", ["d"], "¿Qué tiene agujas y no cose?", "reloj", proposed_by_character="Marlowe",
                            rewards=[ItemReward(description="d", item_name="Llave del Despacho")]))
    outcome = apply_turn(world, _intent(moves=[("Llave del Despacho", "Jugador")]), "dame la llave")

    assert world.player.inventory == []
    assert outcome.kinds() == ["handover_refused", "puzzle_presented"]
    assert world.puzzle_states["Acertijo"] == "proposed"


def test_an_unmet_item_requirement_withholds_until_the_item_is_given():
    world = _world()
    world.characters["Marlowe"].interaction = CharacterInteraction(
        requires=[ItemRequirement(description="que le traigas la gema", item_name="Gema")], relevance_to_objective="r")

    refused = apply_turn(world, _intent(moves=[("Diario", "Jugador")]), "dame el diario")
    apply_turn(world, _intent(moves=[("Gema", "Jugador")]), "tomo la gema")
    given = apply_turn(world, _intent(moves=[("Gema", "Marlowe"), ("Diario", "Jugador")]), "te doy la gema por el diario")

    assert refused.kinds() == ["handover_refused"]
    assert given.kinds() == ["item_given", "item_received"] and _names(world.player.inventory) == ["Diario"]


def test_giving_or_dropping_what_the_player_does_not_have_is_refused():
    world = _world()
    outcome = apply_turn(world, _intent(moves=[("Orbe", "Marlowe"), ("Gema", "Hall")]), "doy y dejo")

    assert outcome.kinds() == ["give_refused", "drop_refused"]


#---- Passages and movement --------------------------------------------------------

def test_a_blocked_passage_opens_only_with_its_item():
    world = _world()
    closed = apply_turn(world, _intent(passages_to_open=[{"destination": "Despacho"}]), "abro el despacho")
    apply_turn(world, _intent(moves=[("Llave del Despacho", "Jugador")]), "le pido la llave")
    opened = apply_turn(world, _intent(passages_to_open=[{"destination": "Despacho"}], go="Despacho"), "abro y entro")

    assert closed.kinds() == ["passage_closed"]
    assert opened.kinds() == ["passage_opened", "moved"] and world.player.location.name == "Despacho"
    assert opened.arrived_at is world.locations["Despacho"] and opened.first_visit


def test_moving_through_a_blocked_or_missing_connection_is_refused():
    world = _world()
    blocked = apply_turn(world, _intent(go="Despacho"), "voy al despacho")
    apply_turn(world, _intent(go="Torre"), "voy a la torre")
    unconnected = apply_turn(world, _intent(go="Despacho"), "voy al despacho")

    assert blocked.kinds() == ["move_refused"] and unconnected.kinds() == ["move_refused"]
    assert world.player.location.name == "Torre"


#---- Puzzles, characters and clues --------------------------------------------------

def _sabio_puzzle(world):
    world.add_puzzle(Puzzle("Enigma", ["d"], "¿Qué se rompe al nombrarlo?", "el silencio", proposed_by_character="Sabio",
                            rewards=[ItemReward(description="d", item_name="Orbe")], accepted_answers=["silencio"]))
    world.locations["Torre"].items.remove(world.items["Orbe"])
    world.characters["Sabio"].inventory.append(world.items["Orbe"])


def test_addressing_a_character_presents_their_puzzle_and_a_later_answer_solves_it():
    world = _world()
    _sabio_puzzle(world)
    apply_turn(world, _intent(go="Torre"), "voy a la torre")

    presented = apply_turn(world, _intent(characters_addressed=["Sabio"]), "hablo con el sabio")
    solved = apply_turn(world, _intent(), "creo que es silencio")  # the model did not list the attempt

    assert presented.kinds() == ["puzzle_presented"]
    assert solved.kinds() == ["puzzle_solved", "item_received"]
    assert _names(world.player.inventory) == ["Orbe"] and world.puzzle_states["Enigma"] == "solved"


def test_a_wrong_answer_is_reported():
    world = _world()
    _sabio_puzzle(world)
    apply_turn(world, _intent(go="Torre", characters_addressed=["Sabio"]), "voy y le hablo")

    outcome = apply_turn(world, _intent(puzzle_attempts=[{"puzzle_name": "Enigma", "answer": "el viento"}]), "es el viento")

    assert outcome.kinds() == ["puzzle_failed"] and world.puzzle_states["Enigma"] == "proposed"


def test_a_puzzle_blocking_a_passage_opens_it_when_solved():
    world = _world()
    hall, torre = world.locations["Hall"], world.locations["Torre"]
    puzzle = Puzzle("Sello", ["d"], "Di la palabra", "abrete")
    world.add_puzzle(puzzle)
    hall.block_passage(torre, puzzle)

    tried = apply_turn(world, _intent(go="Torre"), "voy a la torre")
    solved = apply_turn(world, _intent(go="Torre"), "digo ábrete y paso")

    assert tried.kinds() == ["move_refused", "puzzle_presented"]
    assert "puzzle_solved" in solved.kinds() and world.player.location.name == "Hall"  # moves happen before answers
    assert torre in hall.connecting_locations


def test_addressing_an_absent_character_is_a_fact():
    world = _world()
    outcome = apply_turn(world, _intent(characters_addressed=["Sabio"]), "hablo con el sabio")

    assert outcome.kinds() == ["character_absent"]


def test_examining_an_item_proposes_its_puzzle_and_discovers_its_clue():
    world = _world()
    world.add_puzzle(Puzzle("Grabado", ["d"], "Lee el grabado", "luna", proposed_by_location="Gema"))
    clue = MysteryClue("Brillo", "La gema brilla", "Gema", "r")
    world.objective = (world.player, MysteryObjective("m", "d", [clue], "s"))

    outcome = apply_turn(world, _intent(examined_items=["Gema"]), "examino la gema")

    assert outcome.kinds() == ["clue_discovered", "puzzle_presented"] and clue.discovered


def test_the_outcome_renders_successes_and_failures():
    world = _world()
    outcome = apply_turn(world, _intent(moves=[("Gema", "Jugador"), ("Orbe", "Jugador")]), "tomo")

    rendered = outcome.render("es")

    assert rendered.splitlines()[0].startswith("- ✓ El jugador toma Gema")
    assert rendered.splitlines()[1].startswith("- ✗ Orbe no está al alcance")


def test_a_door_blocked_from_both_sides_opens_from_both_sides():
    world = _world()
    hall, despacho = world.locations["Hall"], world.locations["Despacho"]
    llave = world.items["Llave del Despacho"]
    despacho.block_passage(hall, llave)  # the same door, seen from inside
    world.characters["Marlowe"].inventory.remove(llave)
    world.player.inventory.append(llave)

    apply_turn(world, _intent(passages_to_open=[{"destination": "Despacho"}]), "abro la puerta")

    assert despacho in hall.connecting_locations and hall in despacho.connecting_locations
    assert hall.blocked_locations == {} and despacho.blocked_locations == {}
