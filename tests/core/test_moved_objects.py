"""World.update_from_structured: moving objects between the player, characters and locations."""

import pytest

from src.ivie.core.world import Character, Item, Location, World
from src.ivie.llm.structured_data_models import WorldUpdate


def _world():
    hall, garden = Location("Hall", ["d"]), Location("Jardín", ["d"])
    hall.connecting_locations.append(garden)
    garden.connecting_locations.append(hall)
    card = Item("Moriarty's Calling Card", ["d"])
    lexicon = Item("Ancient Runes Lexicon", ["d"])
    hall.items.append(card)
    player = Character("Dr. John Watson", ["d"], location=hall)
    lyra = Character("Elder Lyra", ["d"], location=hall, inventory=[lexicon])
    world = World(player)
    world.add_locations([hall, garden])
    world.add_items([card, lexicon])
    world.add_character(lyra)
    return world, hall, garden, card, lexicon, lyra


def _move(world, *moves):
    world.update_from_structured(WorldUpdate(
        narration="n", moved_objects=[{"object_name": name, "new_location": to} for name, to in moves]))


@pytest.mark.parametrize("destination", ["Player Inventory", "Player's Inventory", "inventario del jugador",
                                         "dr. john watson", "Watson", "jugador"])
def test_taking_an_item_moves_it_to_the_player_without_duplicates(destination):
    world, hall, garden, card, lexicon, lyra = _world()

    _move(world, ("Moriarty's Calling Card", destination))

    assert world.player.inventory == [card]
    assert hall.items == []


def test_a_character_hands_an_item_to_the_player():
    world, hall, garden, card, lexicon, lyra = _world()

    _move(world, ("Ancient Runes Lexicon", "Player Inventory"))

    assert world.player.inventory == [lexicon]
    assert lyra.inventory == []


def test_giving_uses_a_flexible_character_name():
    world, hall, garden, card, lexicon, lyra = _world()
    _move(world, ("Moriarty's Calling Card", "Player"))

    _move(world, ("Moriarty's Calling Card", "elder lyra"))

    assert card in lyra.inventory and card not in world.player.inventory


def test_an_item_the_player_does_not_hold_is_not_given_or_dropped():
    world, hall, garden, card, lexicon, lyra = _world()

    _move(world, ("Ancient Runes Lexicon", "Jardín"))     # Lyra holds it, not the player
    _move(world, ("Moriarty's Calling Card", "Elder Lyra"))  # lying in the hall

    assert lyra.inventory == [lexicon] and garden.items == []
    assert hall.items == [card] and card not in lyra.inventory


def test_dropping_somewhere_unknown_leaves_the_item_here_once():
    world, hall, garden, card, lexicon, lyra = _world()
    _move(world, ("Moriarty's Calling Card", "Player"))

    _move(world, ("Moriarty's Calling Card", "on the floor"))

    assert hall.items == [card] and world.player.inventory == []


def test_taking_a_blocking_item_does_not_skip_the_other_moves():
    world, hall, garden, card, lexicon, lyra = _world()
    barricade = Item("Barricada", ["d"])
    world.add_item(barricade)
    hall.block_passage(garden, barricade)

    _move(world, ("Barricada", "Player"), ("Moriarty's Calling Card", "Player"))

    assert barricade in world.player.inventory and card in world.player.inventory
    assert garden in hall.connecting_locations
