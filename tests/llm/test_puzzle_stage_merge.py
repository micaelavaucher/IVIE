"""Tests for merge_puzzle_stage: step 4 adds puzzles and blocks without rewriting the world."""

from src.ivie.llm.facet_reconciler import merge_puzzle_stage
from src.ivie.llm.structured_data_models import (
    BlockedPassage,
    BlockedPassageAddition,
    ComponentType,
    GeneratedCharacter,
    GeneratedItem,
    GeneratedLocation,
    GeneratedObjective,
    GeneratedPuzzle,
    GeneratedWorld,
    ItemActionType,
    ItemRequirement,
    ItemReward,
    ObjectiveComponent,
    ObjectiveType,
    PassageReward,
    PuzzleRequirement,
    PuzzleStageResult,
)


def _world(items_in_hall=(), mira_inventory=(), items=("Gem", "Key")):
    return GeneratedWorld(
        locations=[
            GeneratedLocation(name="Hall", descriptions=["d"], items=list(items_in_hall), connecting_locations=["Vault"]),
            GeneratedLocation(name="Vault", descriptions=["d"], connecting_locations=["Hall"]),
        ],
        items=[GeneratedItem(name=name, action_type=ItemActionType.LORE, descriptions=["d"]) for name in items],
        characters=[GeneratedCharacter(name="Mira", descriptions=["d"], location="Hall", inventory=list(mira_inventory))],
        puzzles=[],
        player=GeneratedCharacter(name="Player", descriptions=["d"], location="Hall"),
        objective=GeneratedObjective(
            type=ObjectiveType.REACH_LOCATION,
            components=[ObjectiveComponent(name="Vault", component_type=ComponentType.LOCATION, role_in_objective="goal")],
            description="Reach the vault", success_conditions=["there"]),
        dependency_chains=[], world_theme="t", narrative_context="n",
    )


def _puzzle(name="Riddle", proposer=None, rewards=None, location=None):
    return GeneratedPuzzle(
        name=name, puzzle_type="riddle", descriptions=["d"], problem="¿Qué tiene agujas pero no cose?", answer="reloj",
        location=location, proposed_by_character=proposer, rewards=rewards or [],
        relevance_to_objective="r", hint="h")


def _block(to="Vault", requirement=None):
    return BlockedPassageAddition(from_location="Hall", passage=BlockedPassage(
        location=to, obstacle_name="Door", obstacle_description="Shut",
        required_to_unblock=requirement or PuzzleRequirement(description="solve", puzzle_name="Riddle"),
        relevance_to_objective="r"))


def test_world_is_carried_over_and_puzzles_added():
    world = _world(items_in_hall=["Key"])
    merged, report = merge_puzzle_stage(world, PuzzleStageResult(puzzles=[_puzzle()], blocked_passage_additions=[_block()]))

    assert [p.name for p in merged.puzzles] == ["Riddle"]
    assert [b.location for b in merged.locations[0].blocked_passages] == ["Vault"]
    assert merged.locations[0].items == ["Key"]
    assert world.puzzles == []  # the input world is not mutated


def test_proposes_puzzle_is_derived_from_the_puzzle():
    merged, _ = merge_puzzle_stage(_world(mira_inventory=["Gem"]), PuzzleStageResult(
        puzzles=[_puzzle(proposer="Mira", rewards=[ItemReward(description="d", item_name="Gem")])]))

    mira = merged.characters[0]
    assert mira.interaction is not None
    assert mira.interaction.proposes_puzzle == "Riddle"


def test_reward_lying_elsewhere_moves_to_the_proposer():
    merged, report = merge_puzzle_stage(_world(items_in_hall=["Gem"]), PuzzleStageResult(
        puzzles=[_puzzle(proposer="Mira", rewards=[ItemReward(description="d", item_name="Gem")])]))

    assert merged.characters[0].inventory == ["Gem"]
    assert merged.locations[0].items == []
    assert any("moved to their inventory" in m for m in report.repairs)


def test_extra_puzzles_are_dropped_with_the_blocks_that_need_them():
    stage = PuzzleStageResult(
        puzzles=[_puzzle("Riddle"), _puzzle("Second")],
        blocked_passage_additions=[_block(requirement=PuzzleRequirement(description="s", puzzle_name="Second"))])
    merged, report = merge_puzzle_stage(_world(), stage, max_puzzles=1)

    assert [p.name for p in merged.puzzles] == ["Riddle"]
    assert merged.locations[0].blocked_passages == []


def test_puzzle_in_unknown_location_is_dropped():
    merged, report = merge_puzzle_stage(_world(), PuzzleStageResult(puzzles=[_puzzle(location="Atlantis")]))

    assert merged.puzzles == []
    assert any("non-existent" in m for m in report.repairs)


def test_passage_reward_between_unknown_locations_is_dropped():
    reward = PassageReward(description="d", from_location="Hall", to_location="Atlantis")
    merged, report = merge_puzzle_stage(_world(), PuzzleStageResult(puzzles=[_puzzle(rewards=[reward])]))

    assert merged.puzzles[0].rewards == []


def test_block_with_an_unknown_key_is_dropped():
    stage = PuzzleStageResult(puzzles=[_puzzle()], blocked_passage_additions=[
        _block(requirement=ItemRequirement(description="k", item_name="Skeleton Key"))])
    merged, report = merge_puzzle_stage(_world(), stage)

    assert merged.locations[0].blocked_passages == []
    assert any("would not be enforced" in m for m in report.repairs)


def test_keys_and_rewards_are_made_gettable():
    world = _world(items_in_hall=["Key"], mira_inventory=["Gem"])
    for item in world.items:
        item.gettable = False
    stage = PuzzleStageResult(
        puzzles=[_puzzle(proposer="Mira", rewards=[ItemReward(description="d", item_name="Gem")])],
        blocked_passage_additions=[_block(requirement=ItemRequirement(description="k", item_name="Key"))])
    merged, _ = merge_puzzle_stage(world, stage)

    assert all(item.gettable for item in merged.items)


def test_puzzle_proposed_by_an_item_is_kept():
    puzzle = _puzzle()
    puzzle.proposed_by_location = "Gem"
    merged, _ = merge_puzzle_stage(_world(items_in_hall=["Gem"]), PuzzleStageResult(puzzles=[puzzle]))

    assert merged.puzzles[0].proposed_by_location == "Gem"


def test_proposed_by_something_that_is_not_an_item_is_cleared_not_dropped():
    puzzle = _puzzle()
    puzzle.proposed_by_location = "Vault"   # a location: the engine only matches item names
    merged, report = merge_puzzle_stage(_world(), PuzzleStageResult(puzzles=[puzzle]))

    assert [p.name for p in merged.puzzles] == ["Riddle"]
    assert merged.puzzles[0].proposed_by_location is None
    assert any("not an item" in m for m in report.repairs)


def test_clues_follow_a_reward_moved_to_its_proposer():
    from src.ivie.llm.structured_data_models import MysteryClue

    world = _world(items_in_hall=["Gem"])
    world.objective = GeneratedObjective(
        type=ObjectiveType.SOLVE_MYSTERY,
        components=[ObjectiveComponent(name="Vault", component_type=ComponentType.LOCATION, role_in_objective="g")],
        description="Who did it", success_conditions=["solved"], mystery_solution="Mira",
        mystery_clues=[MysteryClue(name="Shine", description="d", associated_item="Gem", item_location="Hall",
                                   relevance_to_mystery="r")])
    merged, _ = merge_puzzle_stage(world, PuzzleStageResult(
        puzzles=[_puzzle(proposer="Mira", rewards=[ItemReward(description="d", item_name="Gem")])]))

    assert merged.characters[0].inventory == ["Gem"]
    assert merged.objective.mystery_clues[0].item_location is None


def test_a_key_held_by_a_character_with_no_way_to_get_it_becomes_given():
    world = _world(mira_inventory=["Key"])
    stage = PuzzleStageResult(puzzles=[_puzzle()], blocked_passage_additions=[
        _block(requirement=ItemRequirement(description="k", item_name="Key"))])

    merged, report = merge_puzzle_stage(world, stage)

    assert merged.characters[0].interaction.gives_item == "Key"
    assert any("no way to get it" in m for m in report.repairs)


def test_a_key_that_is_the_reward_of_the_holders_puzzle_is_left_alone():
    world = _world(mira_inventory=["Key"])
    stage = PuzzleStageResult(
        puzzles=[_puzzle(proposer="Mira", rewards=[ItemReward(description="d", item_name="Key")])],
        blocked_passage_additions=[_block(requirement=ItemRequirement(description="k", item_name="Key"))])

    merged, _ = merge_puzzle_stage(world, stage)

    assert merged.characters[0].interaction.gives_item is None
    assert merged.characters[0].interaction.proposes_puzzle == "Riddle"


def test_an_item_nobody_needs_is_not_given_away():
    merged, _ = merge_puzzle_stage(_world(mira_inventory=["Gem"]), PuzzleStageResult(puzzles=[_puzzle()]))

    assert merged.characters[0].interaction is None


def test_a_puzzle_nobody_presents_is_given_to_the_character_at_its_location():
    merged, report = merge_puzzle_stage(_world(), PuzzleStageResult(puzzles=[_puzzle(location="Hall")]))

    assert merged.puzzles[0].proposed_by_character == "Mira"
    assert merged.characters[0].interaction.proposes_puzzle == "Riddle"


def test_without_a_character_an_item_at_its_location_presents_it():
    world = _world(items_in_hall=["Gem"])
    world.characters[0].location = "Vault"

    merged, _ = merge_puzzle_stage(world, PuzzleStageResult(puzzles=[_puzzle(location="Hall")]))

    assert merged.puzzles[0].proposed_by_location == "Gem"


def test_dropping_a_puzzle_unblocks_its_passage():
    from src.ivie.llm.facet_reconciler import drop_puzzles
    merged, _ = merge_puzzle_stage(_world(), PuzzleStageResult(
        puzzles=[_puzzle(proposer="Mira")], blocked_passage_additions=[_block()]))

    dropped, report = drop_puzzles(merged, ["Riddle"])

    assert dropped.puzzles == [] and dropped.locations[0].blocked_passages == []
    assert "Vault" in dropped.locations[0].connecting_locations
    assert dropped.characters[0].interaction.proposes_puzzle is None
