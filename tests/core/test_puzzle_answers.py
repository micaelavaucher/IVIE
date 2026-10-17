"""Puzzles are solved by what the player typed, not by the answer the reasoning model reports."""

import pytest

from src.ivie.core.world import Character, Location, Puzzle, World
from src.ivie.llm.structured_data_models import WorldUpdate


def _world(answer):
    hall = Location("Hall", ["d"])
    world = World(Character("Player", ["d"], location=hall))
    world.add_location(hall)
    world.add_puzzle(Puzzle("Riddle", ["d"], "p", answer))
    return world


def _attempt(world, player_input, model_answer, success=True):
    world.update_from_structured(
        WorldUpdate(narration="n", puzzles_solved=[{"puzzle_name": "Riddle", "answer": model_answer, "success": success}]),
        player_input=player_input)
    return world.puzzle_states["Riddle"] == "solved"


@pytest.mark.parametrize("player_input, answer", [
    ("le respondo al Doc: el interruptor correcto es el segundo", "Segundo"),
    ("La respuesta es FLUJO", "flujo"),
    ("respondo: corazon", "Corazón"),
    ("the answer is the Diogenes Club!", "Diogenes Club"),
])
def test_the_answer_inside_a_sentence_solves_the_puzzle(player_input, answer):
    assert _attempt(_world(answer), player_input, model_answer="whatever")


def test_the_model_cannot_answer_for_the_player():
    assert not _attempt(_world("FLUJO"), "examino el Péndulo Desincronizado", model_answer="FLUJO")


def test_part_of_a_word_is_not_the_answer():
    assert not _attempt(_world("sol"), "miro la consola", model_answer="sol")


def test_without_player_input_the_old_comparison_applies():
    world = _world("Segundo")
    world.update_from_structured(WorldUpdate(narration="n", puzzles_solved=[
        {"puzzle_name": "Riddle", "answer": "segundo", "success": True}]))
    assert world.puzzle_states["Riddle"] == "solved"


def test_an_accepted_answer_solves_the_puzzle():
    world = _world("Eco")
    world.puzzles["Riddle"].accepted_answers = ["resonancia", "eco silente"]

    assert _attempt(world, "creo que lo que sos es una resonancia", model_answer="resonancia")


def test_a_part_of_the_answer_is_not_the_answer():
    world = _world("Torre del Reloj")
    world.puzzles["Riddle"].accepted_answers = ["reloj", "la torre del reloj"]

    assert not _attempt(world, "es un reloj de arena", model_answer="reloj")
    assert _attempt(world, "la torre del reloj", model_answer="torre del reloj")


def test_a_puzzle_saved_before_accepted_answers_still_works():
    world = _world("Segundo")
    del world.puzzles["Riddle"].accepted_answers  # an old saved world

    assert _attempt(world, "es el segundo", model_answer="x")


def test_an_accepted_answer_that_keeps_most_of_a_long_answer_solves_the_puzzle():
    world = _world("Primer Fragmento de la Crisálida Madre")
    world.puzzles["Riddle"].accepted_answers = ["fragmento de la crisálida madre", "primer fragmento", "el fragmento"]

    assert _attempt(world, "sos el fragmento de la crisalida madre", model_answer="el fragmento de la crisalida madre")
    assert world.puzzles["Riddle"].answers() == ["Primer Fragmento de la Crisálida Madre", "fragmento de la crisálida madre"]


@pytest.mark.parametrize("answer, fragment", [
    ("Primer Fragmento de la Crisálida Madre", "primer fragmento"),
    ("short-long-short-long", "short long"),
    ("Torre del Reloj", "reloj"),
])
def test_an_accepted_answer_that_is_half_of_the_answer_or_less_does_not_count(answer, fragment):
    world = _world(answer)
    world.puzzles["Riddle"].accepted_answers = [fragment]

    assert not _attempt(world, f"la respuesta es {fragment}", model_answer=fragment)
