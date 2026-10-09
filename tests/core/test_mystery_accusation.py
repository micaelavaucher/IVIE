"""Mysteries are solved by accusing the culprit once every clue is found; failed attempts are facts."""

from src.ivie.core.game_logic import check_objective_completion
from src.ivie.core.turn_engine import apply_turn
from src.ivie.core.world import MysteryClue, MysteryObjective, Puzzle

from tests.core.test_turn_engine import _intent, _world


def _mystery(world, culprit="Marlowe"):
    """One clue, on the Gema lying in the Hall."""
    clue = MysteryClue("Brillo", "La gema tiene una marca de Marlowe", "Gema", "r")
    mystery = MysteryObjective("m", "d", [clue], "Marlowe robó el orbe para venderlo", culprit=culprit,
                               question="¿Quién robó el orbe?")
    world.objective = (world.player, mystery)
    return mystery


#---- Accusations -------------------------------------------------------------------

def test_an_accusation_before_every_clue_lacks_the_evidence():
    world = _world()
    mystery = _mystery(world)

    outcome = apply_turn(world, _intent(accusation="Marlowe"), "fue Marlowe")

    assert outcome.kinds() == ["accusation_premature"] and not world.check_objective()
    assert "0/1" in outcome.render('es')


def test_accusing_someone_else_does_not_solve_the_mystery():
    world = _world()
    _mystery(world)
    apply_turn(world, _intent(examined_items=["Gema"]), "examino la gema")

    outcome = apply_turn(world, _intent(accusation="Sabio"), "fue el Sabio")

    assert outcome.kinds() == ["accusation_failed"] and not world.check_objective()


def test_accusing_the_culprit_with_every_clue_solves_the_mystery():
    world = _world()
    _mystery(world)
    found = apply_turn(world, _intent(examined_items=["Gema"]), "examino la gema")
    assert not world.check_objective()  # every clue is not enough: the culprit must be named

    outcome = apply_turn(world, _intent(accusation="marlowe"), "acuso a Marlowe")

    assert found.kinds() == ["clue_discovered"]
    assert outcome.kinds() == ["mystery_solved"] and world.check_objective()
    assert "Marlowe robó el orbe" in outcome.render('es')  # the narrator writes the ending from it


def test_a_mystery_without_a_culprit_is_solved_by_its_clues():
    world = _world()
    _mystery(world, culprit=None)

    apply_turn(world, _intent(examined_items=["Gema"]), "examino la gema")

    assert world.check_objective()


def test_the_objective_states_the_question_and_the_accusation():
    world = _world()
    _mystery(world)

    statement = world.objective_statement('es')

    assert "¿Quién robó el orbe?" in statement and "responsable" in statement


#---- The ending is announced once ---------------------------------------------------------

def test_the_completion_banner_is_shown_only_on_the_completing_turn():
    world = _world()
    _mystery(world, culprit=None)
    apply_turn(world, _intent(examined_items=["Gema"]), "examino la gema")

    now = check_objective_completion(world, "", 'es', include_completion_text=False, completed_before=False)
    later = check_objective_completion(world, "", 'es', include_completion_text=False, completed_before=True)

    assert "Completaste" in now and later == ""
    assert "Solución" not in now  # the narrator writes the ending, not a pre-written text


#---- Attempts that do not happen are facts too ---------------------------------------------

def test_saying_a_puzzle_is_solved_without_an_answer_is_reported_as_such():
    world = _world()
    world.add_puzzle(Puzzle("Enigma", ["d"], "¿Qué corre sin piernas?", "agua", proposed_by_character="Marlowe"))
    apply_turn(world, _intent(characters_addressed=["Marlowe"]), "hablo con Marlowe")

    outcome = apply_turn(world, _intent(puzzle_attempts=[{"puzzle_name": "Enigma", "answer": ""}]), "ya lo resolví")

    assert outcome.kinds() == ["puzzle_unanswered"] and world.puzzle_states["Enigma"] == "proposed"


def test_using_an_item_the_player_does_not_have_is_reported():
    world = _world()

    missing = apply_turn(world, _intent(items_used=["revólver"]), "disparo con mi revólver")
    elsewhere = apply_turn(world, _intent(items_used=["Orbe"]), "uso el orbe")
    at_hand = apply_turn(world, _intent(items_used=["Gema"]), "uso la gema")

    assert missing.kinds() == ["item_missing"] and "revólver" in missing.render('es')
    assert elsewhere.kinds() == ["item_missing"]
    assert at_hand.kinds() == []


def test_examining_scenery_the_world_does_not_model_is_not_a_failure():
    world = _world()

    outcome = apply_turn(world, _intent(examined_items=["papeles del escritorio"]), "miro los papeles")

    assert outcome.kinds() == []


def test_a_passage_does_not_open_with_another_item():
    world = _world()
    apply_turn(world, _intent(moves=[("Llave del Despacho", "Jugador"), ("Gema", "Jugador")]), "tomo todo")

    with_gem = apply_turn(world, _intent(passages_to_open=[{"destination": "Despacho", "using_item": "Gema"}]), "abro con la gema")
    with_gun = apply_turn(world, _intent(passages_to_open=[{"destination": "Despacho", "using_item": "revólver"}]), "le disparo a la cerradura")
    with_key = apply_turn(world, _intent(passages_to_open=[{"destination": "Despacho", "using_item": "llave del despacho"}]), "abro con la llave")

    assert with_gem.kinds() == ["passage_closed"] and with_gun.kinds() == ["item_missing"]
    assert with_key.kinds() == ["passage_opened"]


def test_an_attempt_reaching_the_engine_twice_is_one_fact():
    world = _world()

    outcome = apply_turn(world, _intent(items_used=["revólver"],
                                        passages_to_open=[{"destination": "Despacho", "using_item": "revólver"}]),
                         "le disparo a la cerradura con mi revólver")

    assert outcome.kinds() == ["item_missing"]
