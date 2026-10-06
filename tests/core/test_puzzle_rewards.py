"""Solving a puzzle must hand over its rewards (RewardType values are lowercase)."""

from src.ivie.core.world import Character, Item, Location, Puzzle, World
from src.ivie.llm.structured_data_models import ItemReward, PassageReward


def _world():
    hall, vault = Location("Hall", ["d"]), Location("Vault", ["d"])
    hall.connecting_locations.append(vault)
    vault.connecting_locations.append(hall)
    gem = Item("Gem", ["d"])
    player = Character("Player", ["d"], location=hall)
    mira = Character("Mira", ["d"], location=hall, inventory=[gem])
    world = World(player)
    world.add_locations([hall, vault])
    world.add_item(gem)
    world.add_character(mira)
    return world, hall, vault, gem, mira


def test_item_reward_moves_from_the_proposer_to_the_player():
    world, hall, vault, gem, mira = _world()
    puzzle = Puzzle("Riddle", ["d"], "p", "luna", proposed_by_character="Mira",
                    rewards=[ItemReward(description="d", item_name="Gem")])
    world.add_puzzle(puzzle)

    world._apply_puzzle_rewards(puzzle)

    assert gem in world.player.inventory
    assert gem not in mira.inventory


def test_passage_reward_unblocks_the_passage():
    world, hall, vault, gem, mira = _world()
    lock = Item("Lock", ["d"], gettable=False)
    hall.block_passage(vault, lock)
    puzzle = Puzzle("Riddle", ["d"], "p", "luna",
                    rewards=[PassageReward(description="d", from_location="Hall", to_location="Vault")])
    world.add_puzzle(puzzle)

    world._apply_puzzle_rewards(puzzle)

    assert vault in hall.connecting_locations
