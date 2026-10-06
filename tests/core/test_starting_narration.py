import time

import pytest

import examples.example_worlds as example_worlds
from src.ivie.core import game_logic


class SlowNarrator:
    model_name = "fake"

    def __init__(self, delay=0.3):
        self.delay = delay

    def prompt_model(self, system_msg, user_msg):
        time.sleep(self.delay)
        if "forma alternativa de narrar el objetivo" in system_msg:
            return "#El objetivo es encontrar la tortuga perdida en el jardín de la casa.#"
        return "Una habitación con pisos de madera."


@pytest.mark.parametrize("parallel", [False, True])
def test_starting_narration_contains_scene_and_objective(monkeypatch, parallel):
    narrator = SlowNarrator()
    monkeypatch.setattr(game_logic, "get_generation_flags",
                        lambda config=None: {"facets": False, "puzzle_stage_only": False, "parallel": parallel})
    monkeypatch.setattr(game_logic, "get_llm", lambda name: SlowNarrator())
    world = example_worlds.get_world("tutorial", language="es")

    started = time.perf_counter()
    narration = game_logic.generate_starting_narration(world, "es", narrator)
    elapsed = time.perf_counter() - started

    assert narration.startswith("Una habitación con pisos de madera.")
    assert "🎯 El objetivo es encontrar la tortuga" in narration
    if parallel:
        assert elapsed < 0.5
    else:
        assert elapsed >= 0.6
