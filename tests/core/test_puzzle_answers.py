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
