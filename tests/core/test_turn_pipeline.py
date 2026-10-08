"""The game loop with NarrateFromOutcome: interpret, apply in the engine, narrate the facts."""

import streamlit as st

from src.ivie.core import game_logic
from src.ivie.core.world import Character, Item, Location, World


class Reasoner:
    """Returns one TurnIntent (as a dict) per turn and records what it was shown."""
    model_name = "fake-reasoner"

    def __init__(self, intents):
        self.intents = list(intents)
        self.prompts = []

    def prompt_model_structured(self, prompt, response_schema, **kwargs):
        self.prompts.append(prompt)
        return self.intents.pop(0)


class Narrator:
    model_name = "fake-narrator"

    def __init__(self):
        self.user_messages = []
        self.system_messages = []

    def prompt_model(self, system_msg, user_msg):
        self.system_messages.append(system_msg)
        self.user_messages.append(user_msg)
        return f"NARRACIÓN {len(self.user_messages)}"


def _world():
    hall = Location("Hall", ["Un hall"])
    key = Item("Llave", ["Una llave"])
    player = Character("Jugador", ["Yo"], location=hall)
    marlowe = Character("Marlowe", ["Un detective"], location=hall, inventory=[key])
    world = World(player)
    world.add_location(hall)
    world.add_item(key)
    world.add_character(marlowe)
    world.set_objective(player, key)
    world.player.location.visited = True
    return world


def _loop(world, reasoner, narrator):
    st.session_state["nickname"] = "test"
    return game_logic.create_game_loop(world, reasoner, narrator, "es", set(), enable_rag=False, narrate_from_outcome=True)


def _intent(**fields):
    return {"moved_objects": [], "location_changed": {"new_location": None}, "passages_to_open": [],
            "examined_items": [], "characters_addressed": [], "puzzle_attempts": [], "action_summary": "s", **fields}


def test_the_narration_is_written_from_the_engine_facts():
    world, narrator = _world(), Narrator()
    reasoner = Reasoner([_intent(characters_addressed=["Marlowe"]),
                         _intent(moved_objects=[{"object_name": "Llave", "new_location": "Jugador"}])])
    loop = _loop(world, reasoner, narrator)

    first = loop("hola Marlowe", [])
    second = loop("Marlowe, dame la llave", [])

    assert first.startswith("NARRACIÓN 1") and "Completaste" not in first
    assert "HECHOS DEL MOTOR" in narrator.user_messages[1]
    assert "✓ Marlowe le entrega Llave al jugador" in narrator.user_messages[1]
    assert "El objetivo NO está cumplido" in narrator.system_messages[0]
    assert "ACABA DE CUMPLIR" in narrator.system_messages[1]
    assert second.startswith("NARRACIÓN 2") and "Completaste el objetivo" in second


def test_the_reasoner_does_not_see_puzzle_answers():
    from src.ivie.core.world import Puzzle
    world = _world()
    world.add_puzzle(Puzzle("Acertijo", ["d"], "¿Qué es?", "respuesta-secreta", proposed_by_character="Marlowe"))
    reasoner, narrator = Reasoner([_intent(characters_addressed=["Marlowe"])]), Narrator()

    _loop(world, reasoner, narrator)("hablo con Marlowe", [])

    assert "¿Qué es?" in narrator.user_messages[0]
    assert "respuesta-secreta" not in reasoner.prompts[0] and "respuesta-secreta" not in narrator.user_messages[0]


def test_a_failed_interpretation_changes_nothing():
    world, narrator = _world(), Narrator()

    class Broken(Reasoner):
        def prompt_model_structured(self, prompt, response_schema, **kwargs):
            raise RuntimeError("boom")

    reply = _loop(world, Broken([]), narrator)("Marlowe, dame la llave", [])

    assert reply.startswith("NARRACIÓN 1") and world.player.inventory == []
    assert "ningún cambio" in narrator.user_messages[0]
