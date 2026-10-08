"""Opening narration, objective sentence and hint commands."""

import pytest

import examples.example_worlds as example_worlds
from src.ivie.core import game_logic
from src.ivie.core.world import Character, Item, Location, MysteryClue, MysteryObjective, World


class Narrator:
    model_name = "fake"

    def __init__(self):
        self.calls = 0

    def prompt_model(self, system_msg, user_msg):
        self.calls += 1
        return "Una habitación con pisos de madera."


def test_opening_narration_has_the_scene_and_the_objective_sentence():
    narrator = Narrator()
    world = example_worlds.get_world("tutorial", language="es")

    narration = game_logic.generate_starting_narration(world, "es", narrator)

    assert narration.startswith("Una habitación con pisos de madera.")
    assert "🎯 Tienes que conseguir Tortuga." in narration
    assert narrator.calls == 1  # the objective sentence needs no model call


def _world():
    hall, tower = Location("Hall", ["d"]), Location("Torre", ["d"])
    player = Character("Jugador", ["d"], location=hall)
    orb = Item("Orbe Armónico", ["d"])
    sage = Character("Sabio", ["d"], location=tower)
    world = World(player)
    world.add_locations([hall, tower])
    world.add_item(orb)
    world.add_character(sage)
    return world, player, tower, orb, sage


@pytest.mark.parametrize("objective, es, en", [
    (lambda p, t, o, s: (p, o), "Tienes que conseguir Orbe Armónico.", "You have to get Orbe Armónico."),
    (lambda p, t, o, s: (p, t), "Tienes que llegar a Torre.", "You have to reach Torre."),
    (lambda p, t, o, s: (p, s), "Tienes que encontrar a Sabio.", "You have to find Sabio."),
    (lambda p, t, o, s: (o, t), "Tienes que llevar Orbe Armónico a Torre.", "You have to take Orbe Armónico to Torre."),
    (lambda p, t, o, s: (o, s), "Tienes que entregarle Orbe Armónico a Sabio.", "You have to give Orbe Armónico to Sabio."),
])
def test_objective_sentence_names_the_real_elements(objective, es, en):
    world, player, tower, orb, sage = _world()
    world.objective = objective(player, tower, orb, sage)

    assert world.objective_statement("es") == es
    assert world.objective_statement("en") == en


def test_mystery_sentence_states_the_real_number_of_clues():
    world, player, tower, orb, sage = _world()
    clues = [MysteryClue(f"c{i}", "d", "Orbe Armónico", "r") for i in range(3)]
    world.objective = (player, MysteryObjective("m", "five clues, says the story", clues, "s"))

    assert world.objective_statement("es") == "Tienes que resolver el misterio: descubre las 3 pistas."


@pytest.mark.parametrize("message", ["pista", "Ayuda", "pido una pista", "¡Dame una pista!", "help", "I need a hint"])
def test_hint_commands_are_recognized(message):
    world = example_worlds.get_world("tutorial", language="es")
    world.update_hints()

    assert game_logic.handle_debug_command(message, world, "es").startswith("💡")


@pytest.mark.parametrize("message", ["examino la Pista del Zafiro Roto", "le pido ayuda a Marlowe con la llave"])
def test_actions_that_mention_a_hint_word_are_not_commands(message):
    world = example_worlds.get_world("tutorial", language="es")

    assert game_logic.handle_debug_command(message, world, "es") is None


def test_default_hints_follow_the_game_language():
    world = example_worlds.get_world("tutorial", language="es")
    world.current_hints = [dict(hint) for hint in world.default_explore_hints]

    assert world.get_next_hint("es") == "Explora los lugares a los que puedes ir."
    assert world.get_next_hint("en") == "Talk to the characters and examine the objects around you."


def test_objective_command_shows_the_mystery_progress():
    world, player, tower, orb, sage = _world()
    clues = [MysteryClue("Huella", "d", "Orbe Armónico", "r"), MysteryClue("Carta", "d", "Orbe Armónico", "r")]
    clues[0].discovered = True
    world.objective = (player, MysteryObjective("m", "Encuentra las cinco pistas.", clues, "s"))

    info = game_logic.get_objective_info(world, "es")

    assert info.startswith("🎯 Tienes que resolver el misterio: descubre las 2 pistas.")
    assert "1/2 pistas descubiertas" in info
