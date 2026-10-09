"""narration_check: a narration's claims are checked against the state before and after the turn."""

from src.ivie.core.narration_check import Snapshot, check_claims
from src.ivie.llm.structured_data_models import NarrationClaims

from tests.core.test_turn_pipeline import Narrator, Reasoner, _intent, _loop, _world

ITEMS = {"Informe de la Escena", "Diario Desgastado", "Nota Cifrada", "Llave"}


def _check(before, after, judged=False, completed=False, **claims):
    return check_claims(NarrationClaims(**claims), before, after, ITEMS, answer_judged=judged,
                        objective_completed=completed, language='es')


def test_a_hand_over_the_engine_did_not_make_is_reported():
    here = Snapshot(held_here={"Informe de la Escena"})

    problems = _check(here, here, items_player_receives=["el informe de la escena"])

    assert problems == ["El jugador no recibe Informe de la Escena en este turno: lo sigue teniendo un personaje."]


def test_an_item_really_received_may_be_described_in_other_words():
    before, after = Snapshot(held_here={"Nota Cifrada"}), Snapshot(inventory={"Nota Cifrada"})

    assert _check(before, after, items_player_receives=["un papel arrugado"]) == []
    assert _check(before, after, items_player_receives=["Nota Cifrada", "un pañuelo bordado"]) == [
        "El jugador no recibe 'un pañuelo bordado': no existe en el mundo."]


def test_examining_is_not_taking():
    here = Snapshot(lying_here={"Diario Desgastado"})

    problems = _check(here, here, items_player_receives=["Diario Desgastado"])

    assert problems == ["El jugador no recibe Diario Desgastado en este turno: sigue en el lugar, sin tomar."]


def test_an_item_given_away_in_the_turn_was_at_hand():
    before, after = Snapshot(inventory={"Llave"}), Snapshot()

    assert _check(before, after, items_player_holds_or_uses=["la llave"]) == []
    assert _check(Snapshot(), Snapshot(), items_player_holds_or_uses=["un revólver"]) == [
        "El jugador no tiene un revólver: no está en su inventario."]


def test_an_offer_of_something_nobody_has_is_reported():
    problems = _check(Snapshot(), Snapshot(), items_characters_offer=["el informe oficial"])

    assert problems == ["Ningún personaje tiene 'el informe oficial': no existe en el mundo."]


def test_judging_an_answer_or_celebrating_needs_the_engine():
    assert _check(Snapshot(), Snapshot(), answer_judged=True, objective_completed=True) == [
        "El jugador no dio ninguna respuesta que el juego evaluara: no digas si es correcta o incorrecta.",
        "El objetivo no está cumplido: la aventura sigue."]
    assert _check(Snapshot(), Snapshot(), judged=True, completed=True, answer_judged=True, objective_completed=True) == []


def test_a_narration_that_contradicts_the_world_is_rewritten_once():
    world, narrator = _world(), Narrator()
    invented = {"items_player_receives": ["Llave"]}  # Marlowe never handed it over
    reasoner = Reasoner([_intent(characters_addressed=["Marlowe"])], claims=[invented])
    loop = _loop(world, reasoner, narrator)

    answer = loop("hola Marlowe", [])

    assert answer.startswith("NARRACIÓN 2") and len(narrator.user_messages) == 2
    assert "CONTRADECÍA EL JUEGO" in narrator.user_messages[1]
    assert "El jugador no recibe Llave en este turno" in narrator.user_messages[1]


def test_the_narrator_does_not_see_what_characters_give():
    from src.ivie.llm.structured_data_models import CharacterInteraction
    world = _world()
    world.characters["Marlowe"].interaction = CharacterInteraction(
        gives_item="Llave", interaction_text="Tomá la llave", relevance_to_objective="r")

    for_reasoner = world.render_world(language='es', reveal_answers=False)
    for_narrator = world.render_world(language='es', reveal_answers=False, for_narrator=True)

    assert "Entrega al jugador: Llave" in for_reasoner
    assert "Entrega al jugador" not in for_narrator and "decirlo no entrega nada" in for_narrator
