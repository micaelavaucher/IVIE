"""Objectives that need several items together."""

from src.ivie.core import game_logic
from src.ivie.core.world import Character, Item, ItemCollection, Location, World
from src.ivie.core.world_builder import create_world_from_llm_response, validate_objective_components
from src.ivie.llm.structured_data_models import (
    GeneratedCharacter, GeneratedItem, GeneratedLocation, GeneratedObjective, GeneratedWorld, ObjectiveComponent,
)


def _generated(objective_type, components):
    return GeneratedWorld(
        locations=[GeneratedLocation(name="Hall", descriptions=["d"], items=["Norte", "Sur", "Este"], connecting_locations=["Altar"]),
                   GeneratedLocation(name="Altar", descriptions=["d"], connecting_locations=["Hall"])],
        items=[GeneratedItem(name=n, action_type="lore", descriptions=["d"]) for n in ("Norte", "Sur", "Este")],
        characters=[GeneratedCharacter(name="Sabio", descriptions=["d"], location="Altar")],
        puzzles=[], player=GeneratedCharacter(name="Jugador", descriptions=["d"], location="Hall"),
        objective=GeneratedObjective(
            type=objective_type, description="Reúne los tres fragmentos", success_conditions=["x"],
            components=[ObjectiveComponent(name=n, component_type=t, role_in_objective="g") for n, t in components]),
        dependency_chains=[], world_theme="t", narrative_context="n")


FRAGMENTS = [("Norte", "item"), ("Sur", "item"), ("Este", "item")]


def test_gathering_three_items_needs_all_three():
    world = create_world_from_llm_response(_generated("get_item", FRAGMENTS))
    hall = world.locations["Hall"]

    assert isinstance(world.objective[1], ItemCollection)
    assert world.objective_statement("es") == "Tienes que conseguir Norte, Sur y Este."
    world.player.save_item(world.items["Norte"], hall)
    world.player.save_item(world.items["Sur"], hall)
    assert not world.check_objective()
    world.player.save_item(world.items["Este"], hall)
    assert world.check_objective()
    assert validate_objective_components(world)[0]


def test_delivering_several_items_needs_all_of_them_at_the_recipient():
    world = create_world_from_llm_response(_generated("deliver_an_item", FRAGMENTS + [("Sabio", "character")]))
    sabio = world.characters["Sabio"]

    assert world.objective_statement("en") == "You have to give Norte, Sur and Este to Sabio."
    sabio.inventory += [world.items["Norte"], world.items["Sur"]]
    assert not world.check_objective()
    sabio.inventory.append(world.items["Este"])
    assert world.check_objective()


def test_objective_info_shows_the_collection_progress():
    world = create_world_from_llm_response(_generated("get_item", FRAGMENTS))
    world.player.save_item(world.items["Sur"], world.locations["Hall"])

    info = game_logic.get_objective_info(world, "es")

    assert info.startswith("🎯 Tienes que conseguir Norte, Sur y Este.") and "Progreso: 1/3" in info


def test_a_single_item_objective_is_unchanged():
    world = create_world_from_llm_response(_generated("get_item", [("Norte", "item")]))

    assert world.objective[1] is world.items["Norte"]


def test_the_prewritten_ending_is_left_out_when_the_narrator_writes_it():
    hall = Location("Hall", ["d"])
    key = Item("Llave", ["d"])
    world = World(Character("J", ["d"], location=hall, inventory=[key]))
    world.add_location(hall)
    world.add_item(key)
    world.set_objective(world.player, key)

    class Objective:
        type = "get_item"
        completion_narration = "Con el condensador reprogramado..."

    world.objective_data = Objective()

    assert "reprogramado" in game_logic.check_objective_completion(world, "", "es")
    without = game_logic.check_objective_completion(world, "", "es", include_completion_text=False)
    assert "Completaste el objetivo" in without and "reprogramado" not in without
