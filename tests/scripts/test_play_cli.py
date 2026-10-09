"""scripts/play_cli.py: sessions survive between processes and turns go through the real game loop."""

import importlib.util
import json
import os

import pytest

import examples.example_worlds as example_worlds

_spec = importlib.util.spec_from_file_location(
    "play_cli", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "play_cli.py"))
play_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(play_cli)


class FakeReasoning:
    """Returns one WorldUpdate per turn, in order."""
    model_name = "fake-reasoning"

    def __init__(self, updates):
        self.updates = list(updates)

    def prompt_model_structured(self, prompt, response_schema, **kwargs):
        if response_schema.__name__ == "NarrationClaims":
            return {}
        return self.updates.pop(0)


class FakeNarrator:
    model_name = "fake-narrator"

    def prompt_model(self, system_msg, user_msg):
        return "Llegás a un jardín tranquilo."


def _update(narration, new_location=None, moved=None):
    """A TurnIntent as the reasoning model returns it."""
    return {"action_summary": narration, "moved_objects": moved or [], "passages_to_open": [],
            "location_changed": {"new_location": new_location}, "examined_items": [],
            "characters_addressed": [], "puzzle_attempts": []}


@pytest.fixture
def session(tmp_path, monkeypatch):
    from src.ivie.core import game_logic
    # The CLI plays the turn the app uses; pin it instead of reading config.ini
    monkeypatch.setattr(game_logic, "narrate_from_outcome_enabled", lambda config=None: True)
    monkeypatch.setattr(play_cli, "PLAYTESTS", str(tmp_path))
    name = "test-session"
    session_dir = tmp_path / name
    session_dir.mkdir()
    (session_dir / "meta.json").write_text(json.dumps(
        {"session": name, "language": "es", "theme": "", "mode": "generate", "world_id": "playtest_test",
         "turns": 0, "completed": False}), encoding="utf-8")
    play_cli.save_world(str(session_dir), example_worlds.get_world("tutorial", language="es"))
    return name, session_dir


def test_world_survives_a_save_and_load(session):
    name, session_dir = session

    world = play_cli.load_world(str(session_dir))

    assert world.player.location.name == "Habitación Inicial"
    assert "Objetivo cumplido: False" in play_cli.state_report(world, 0)


def test_turns_are_played_through_the_game_loop_and_saved(session, capsys):
    name, session_dir = session
    reasoning = FakeReasoning([
        _update("Caminás hacia el jardín.", new_location="Jardín"),
        _update("Tomás la tortuga.", moved=[{"object_name": "Tortuga", "new_location": "Jugador"}]),
    ])

    play_cli.play_turn(name, "voy al jardín", reasoning, FakeNarrator(), enable_rag=False)
    play_cli.play_turn(name, "agarro la tortuga", reasoning, FakeNarrator(), enable_rag=False)

    world = play_cli.load_world(str(session_dir))
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert world.player.location.name == "Jardín"
    assert [item.name for item in world.player.inventory] == ["Tortuga"]
    assert meta["turns"] == 2 and meta["completed"] is True
    assert "OBJETIVO COMPLETADO" in output
    transcript = (session_dir / "transcript.md").read_text(encoding="utf-8")
    assert "> voy al jardín" in transcript and "> agarro la tortuga" in transcript
    assert "Engine facts" in (session_dir / "engine.log").read_text(encoding="utf-8")


def test_an_engine_exception_is_reported_not_raised(session, capsys):
    name, session_dir = session

    class BrokenNarrator(FakeNarrator):
        def prompt_model(self, system_msg, user_msg):
            raise RuntimeError("boom again")

    play_cli.play_turn(name, "miro", FakeReasoning([_update("mira")]), BrokenNarrator(), enable_rag=False)

    assert "ERROR DEL MOTOR" in capsys.readouterr().out
    assert "boom again" in (session_dir / "transcript.md").read_text(encoding="utf-8")
