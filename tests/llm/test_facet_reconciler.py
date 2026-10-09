"""Tests for the deterministic merge of independently generated world facets.

Ported from ivie-msc (tests/llm/test_facet_reconciler.py). Left out: the relationship test
(IVIE has no npc_relationships) and the two tests on restricting BlockedPassage's schema (IVIE
keeps the shared schema so the original pipeline is unchanged; unenforceable blocks are dropped
by the reconciler instead, which the remaining tests cover). The default objective of the
helpers targets the LAST location, so the IVIE-specific trivial-objective rule does not fire in
tests that are not about it.

The tests after "IVIE-specific rules" cover what was added on top of the original reconciler.
"""

import pytest

from src.ivie.llm.facet_reconciler import reconcile_world_facets
from src.ivie.llm.structured_data_models import (
    BlockedPassage,
    CharacterInteraction,
    ComponentType,
    GeneratedCharacter,
    GeneratedItem,
    GeneratedLocation,
    GeneratedObjective,
    ItemActionType,
    ItemRequirement,
    MysteryClue,
    ObjectiveComponent,
    ObjectiveType,
    WorldCharactersFacet,
    WorldItemsFacet,
    WorldLocationsFacet,
    WorldObjectiveFacet,
)


#---- Builders ---------------------------------------------------------------

def _location(name, items=None, connecting=None, blocked=None):
    return GeneratedLocation(
        name=name,
        descriptions=[f"{name} description"],
        items=items or [],
        connecting_locations=connecting or [],
        blocked_passages=blocked or [],
    )


def _item(name, action_type=ItemActionType.LORE, gettable=True):
    return GeneratedItem(name=name, action_type=action_type, descriptions=[f"{name} description"],
                         gettable=gettable)


def _character(name, location, inventory=None, interaction=None):
    return GeneratedCharacter(
        name=name,
        descriptions=[f"{name} description"],
        location=location,
        inventory=inventory or [],
        interaction=interaction,
    )


def _objective(objective_type, components, **extra):
    return GeneratedObjective(
        type=objective_type,
        components=[ObjectiveComponent(name=name, component_type=kind, role_in_objective="goal")
                    for name, kind in components],
        description="The objective",
        success_conditions=["Done"],
        **extra,
    )


def _facets(locations, items, characters, player, objective=None):
    return (
        WorldLocationsFacet(locations=locations),
        WorldItemsFacet(items=items),
        WorldCharactersFacet(characters=characters, player=player),
        WorldObjectiveFacet(
            objective=objective or _objective(
                ObjectiveType.REACH_LOCATION, [(locations[-1].name, ComponentType.LOCATION)]),
            world_theme="Testing",
            narrative_context="A test world",
        ),
    )


def _reconcile(locations, items, characters, player, objective=None):
    return reconcile_world_facets(*_facets(locations, items, characters, player, objective))


def _door(to, key):
    return BlockedPassage(
        location=to, obstacle_name="Iron Door", obstacle_description="Heavy",
        required_to_unblock=ItemRequirement(description="Need it", item_name=key),
        relevance_to_objective="the way on",
    )


def _by_name(elements, name):
    return next(element for element in elements if element.name == name)


#---- The headline rule: character inventory wins ----------------------------

def test_character_inventory_wins_over_location_placement():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Brass Key"])],
        items=[_item("Brass Key", ItemActionType.UNLOCK_PASSAGE)],
        characters=[_character("Mira", "Hall", inventory=["Brass Key"])],
        player=_character("Player", "Hall"),
    )

    assert _by_name(world.characters, "Mira").inventory == ["Brass Key"]
    assert world.locations[0].items == []
    assert any("character inventory wins" in m for m in report.conflicts)


def test_item_held_by_two_characters_stays_with_the_first():
    world, report = _reconcile(
        locations=[_location("Hall")],
        items=[_item("Lantern")],
        characters=[
            _character("Mira", "Hall", inventory=["Lantern"]),
            _character("Tomas", "Hall", inventory=["Lantern"]),
        ],
        player=_character("Player", "Hall"),
    )

    holders = [c.name for c in world.characters if "Lantern" in c.inventory]
    assert holders == ["Mira"]
    assert len(report.conflicts) == 1


def test_player_inventory_outranks_an_npc_holding_the_same_item():
    world, report = _reconcile(
        locations=[_location("Hall")],
        items=[_item("Locket")],
        characters=[_character("Mira", "Hall", inventory=["Locket"])],
        player=_character("Player", "Hall", inventory=["Locket"]),
    )

    assert world.player.inventory == ["Locket"]
    assert _by_name(world.characters, "Mira").inventory == []
    assert any("kept with the player" in m for m in report.conflicts)


def test_item_placed_in_two_locations_stays_in_the_first():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Coin"]), _location("Cellar", items=["Coin"])],
        items=[_item("Coin")],
        characters=[],
        player=_character("Player", "Hall"),
    )

    assert world.locations[0].items == ["Coin"]
    assert world.locations[1].items == []
    assert len(report.conflicts) == 1


#---- Repairs: a facet referencing something that does not exist -------------

def test_orphan_item_is_placed_at_the_player_start():
    world, report = _reconcile(
        locations=[_location("Hall"), _location("Cellar")],
        items=[_item("Forgotten Ring")],
        characters=[],
        player=_character("Player", "Cellar"),
        objective=_objective(ObjectiveType.REACH_LOCATION, [("Hall", ComponentType.LOCATION)]),
    )

    assert _by_name(world.locations, "Cellar").items == ["Forgotten Ring"]
    assert any("was placed nowhere" in m for m in report.repairs)


def test_unknown_item_references_are_dropped():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Ghost Item"])],
        items=[_item("Coin")],
        characters=[_character("Mira", "Hall", inventory=["Phantom Blade"])],
        player=_character("Player", "Hall"),
    )

    assert "Ghost Item" not in world.locations[0].items
    assert _by_name(world.characters, "Mira").inventory == []
    assert len(report.repairs) >= 2


def test_character_in_unknown_location_moves_to_player_start():
    world, report = _reconcile(
        locations=[_location("Hall"), _location("Cellar")],
        items=[],
        characters=[_character("Mira", "Nowhere")],
        player=_character("Player", "Cellar"),
        objective=_objective(ObjectiveType.REACH_LOCATION, [("Hall", ComponentType.LOCATION)]),
    )

    assert _by_name(world.characters, "Mira").location == "Cellar"
    assert any("non-existent location" in m for m in report.repairs)


def test_player_in_unknown_location_falls_back_to_the_first_location():
    world, report = _reconcile(
        locations=[_location("Hall"), _location("Cellar")],
        items=[],
        characters=[],
        player=_character("Player", "Atlantis"),
    )

    assert world.player.location == "Hall"
    assert any("player start location" in m for m in report.repairs)


#---- Topology --------------------------------------------------------------

def test_one_way_connections_are_made_symmetric():
    world, report = _reconcile(
        locations=[_location("Hall", connecting=["Cellar"]), _location("Cellar")],
        items=[],
        characters=[],
        player=_character("Player", "Hall"),
    )

    assert _by_name(world.locations, "Cellar").connecting_locations == ["Hall"]
    assert any("one-way" in m for m in report.repairs)


def test_dangling_and_self_connections_are_dropped():
    world, report = _reconcile(
        locations=[_location("Hall", connecting=["Hall", "Atlantis"])],
        items=[],
        characters=[],
        player=_character("Player", "Hall"),
    )

    assert world.locations[0].connecting_locations == []
    assert any("connects to itself" in m for m in report.repairs)
    assert any("non-existent" in m for m in report.repairs)


def test_blocked_passage_destination_is_added_to_connections():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Brass Key"], blocked=[_door("Vault", "Brass Key")]), _location("Vault")],
        items=[_item("Brass Key", ItemActionType.UNLOCK_PASSAGE)],
        characters=[],
        player=_character("Player", "Hall"),
    )

    assert "Vault" in _by_name(world.locations, "Hall").connecting_locations
    assert "Hall" in _by_name(world.locations, "Vault").connecting_locations
    assert any("missing from connections" in m for m in report.repairs)


def test_blocked_passage_to_unknown_location_is_dropped():
    world, report = _reconcile(
        locations=[_location("Hall", blocked=[_door("Atlantis", "Brass Key")])],
        items=[_item("Brass Key")],
        characters=[],
        player=_character("Player", "Hall"),
    )

    assert world.locations[0].blocked_passages == []
    assert any("non-existent" in m for m in report.repairs)


#---- Assembly --------------------------------------------------------------

def test_clean_facets_need_no_changes():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Coin"], connecting=["Cellar"]),
                   _location("Cellar", connecting=["Hall"])],
        items=[_item("Coin")],
        characters=[_character("Mira", "Cellar")],
        player=_character("Player", "Hall"),
    )

    assert report.total == 0
    assert world.world_theme == "Testing"
    assert world.narrative_context == "A test world"
    assert world.dependency_chains == []
    assert world.puzzles == []


def test_facets_are_not_mutated_by_reconciliation():
    locations_facet, items_facet, characters_facet, objective_facet = _facets(
        locations=[_location("Hall", items=["Brass Key"])],
        items=[_item("Brass Key")],
        characters=[_character("Mira", "Hall", inventory=["Brass Key"])],
        player=_character("Player", "Hall"),
    )

    reconcile_world_facets(locations_facet, items_facet, characters_facet, objective_facet)

    # The raw facet output stays intact, so a caller can still log or compare it.
    assert locations_facet.locations[0].items == ["Brass Key"]


def test_puzzles_are_carried_through_when_supplied():
    facets = _facets(locations=[_location("Hall")], items=[], characters=[], player=_character("Player", "Hall"))

    world, _ = reconcile_world_facets(*facets, puzzles=[])

    assert world.puzzles == []


def test_empty_locations_facet_is_a_hard_failure():
    locations_facet, items_facet, characters_facet, objective_facet = _facets(
        locations=[_location("Hall")], items=[], characters=[], player=_character("Player", "Hall"),
    )
    locations_facet.locations = []

    with pytest.raises(ValueError, match="no locations"):
        reconcile_world_facets(locations_facet, items_facet, characters_facet, objective_facet)


#---- Blocks the engine would silently ignore --------------------------------

def test_a_block_requiring_a_nonexistent_item_is_dropped():
    world, report = _reconcile(
        locations=[_location("Hall", blocked=[_door("Vault", "Primary Key")]), _location("Vault")],
        items=[_item("Coin")],          # no Primary Key anywhere
        characters=[],
        player=_character("Player", "Hall"),
    )

    assert world.locations[0].blocked_passages == []
    assert any("would not be enforced" in m and "Primary Key" in m for m in report.repairs)


def test_a_block_requiring_a_real_item_is_kept():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Brass Key"], blocked=[_door("Vault", "Brass Key")]), _location("Vault")],
        items=[_item("Brass Key", ItemActionType.UNLOCK_PASSAGE)],
        characters=[],
        player=_character("Player", "Hall"),
    )

    assert [b.location for b in world.locations[0].blocked_passages] == ["Vault"]


def test_a_puzzle_requirement_cannot_be_enforced_at_the_facet_stage():
    """No puzzles exist yet when facets merge, so a puzzle-gated block cannot resolve here."""
    from src.ivie.llm.structured_data_models import PuzzleRequirement

    blocked = BlockedPassage(
        location="Vault", obstacle_name="Sealed Door", obstacle_description="Sealed",
        required_to_unblock=PuzzleRequirement(description="Solve it", puzzle_name="The Riddle"),
        relevance_to_objective="the way on",
    )
    world, report = _reconcile(
        locations=[_location("Hall", blocked=[blocked]), _location("Vault")],
        items=[], characters=[], player=_character("Player", "Hall"),
    )

    assert world.locations[0].blocked_passages == []
    assert any("The Riddle" in m for m in report.repairs)


def test_a_character_requirement_block_is_dropped():
    from src.ivie.llm.structured_data_models import CharacterRequirement

    blocked = BlockedPassage(
        location="Vault", obstacle_name="Guard", obstacle_description="A guard",
        required_to_unblock=CharacterRequirement(description="Talk to her", character_name="Mira"),
        relevance_to_objective="the way on",
    )
    world, report = _reconcile(
        locations=[_location("Hall", blocked=[blocked]), _location("Vault")],
        items=[], characters=[_character("Mira", "Hall")], player=_character("Player", "Hall"),
    )

    assert world.locations[0].blocked_passages == []
    assert any("not enforced by the engine" in m for m in report.repairs)


#---- Mystery clues -----------------------------------------------------------

def _mystery(clues):
    return _objective(ObjectiveType.SOLVE_MYSTERY, [("Hall", ComponentType.LOCATION)],
                      mystery_clues=clues, mystery_solution="The butler")


def _clue(name, item_name, location=None):
    return MysteryClue(name=name, description="d", associated_item=item_name, item_location=location,
                       relevance_to_mystery="r")


def test_clue_location_is_corrected_to_where_the_item_is():
    world, report = _reconcile(
        locations=[_location("Hall", connecting=["Cellar"]), _location("Cellar", items=["Letter"])],
        items=[_item("Letter")],
        characters=[],
        player=_character("Player", "Hall"),
        objective=_mystery([_clue("Ink stain", "Letter", "Hall")]),
    )

    assert world.objective.mystery_clues[0].item_location == "Cellar"
    assert any("corrected" in m for m in report.repairs)


def test_clue_location_is_cleared_when_a_character_holds_the_item():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Letter"])],
        items=[_item("Letter")],
        characters=[_character("Butler", "Hall", inventory=["Letter"])],
        player=_character("Player", "Hall"),
        objective=_mystery([_clue("Ink stain", "Letter", "Hall")]),
    )

    assert world.objective.mystery_clues[0].item_location is None
    assert any("cleared item_location" in m for m in report.repairs)


def test_clue_item_in_the_player_inventory_is_moved_into_the_world():
    world, report = _reconcile(
        locations=[_location("Hall")],
        items=[_item("Letter")],
        characters=[],
        player=_character("Player", "Hall", inventory=["Letter"]),
        objective=_mystery([_clue("Ink stain", "Letter")]),
    )

    assert world.player.inventory == []
    assert world.locations[0].items == ["Letter"]


#---- IVIE-specific rules ----------------------------------------------------

def test_orphan_objective_item_goes_to_the_farthest_location():
    world, report = _reconcile(
        locations=[_location("Hall", connecting=["Cellar"]), _location("Cellar", connecting=["Vault"]),
                   _location("Vault")],
        items=[_item("Crown"), _item("Coin")],
        characters=[],
        player=_character("Player", "Hall"),
        objective=_objective(ObjectiveType.GET_ITEM, [("Crown", ComponentType.ITEM)]),
    )

    assert _by_name(world.locations, "Vault").items == ["Crown"]
    assert _by_name(world.locations, "Hall").items == ["Coin"]
    assert any("farthest" in m for m in report.repairs)


def test_player_starting_at_the_destination_is_moved():
    world, report = _reconcile(
        locations=[_location("Hall", connecting=["Cellar"]), _location("Cellar", connecting=["Vault"]),
                   _location("Vault")],
        items=[],
        characters=[],
        player=_character("Player", "Vault"),
        objective=_objective(ObjectiveType.REACH_LOCATION, [("Vault", ComponentType.LOCATION)]),
    )

    assert world.player.location == "Hall"
    assert any("objective destination" in m for m in report.repairs)


def test_player_starting_next_to_the_character_to_find_is_moved():
    world, report = _reconcile(
        locations=[_location("Hall", connecting=["Cellar"]), _location("Cellar")],
        items=[],
        characters=[_character("Explorer", "Hall")],
        player=_character("Player", "Hall"),
        objective=_objective(ObjectiveType.FIND_CHARACTER, [("Explorer", ComponentType.CHARACTER)]),
    )

    assert world.player.location == "Cellar"


def test_objective_item_in_the_player_inventory_is_moved_away():
    world, report = _reconcile(
        locations=[_location("Hall", connecting=["Cellar"]), _location("Cellar")],
        items=[_item("Crown")],
        characters=[],
        player=_character("Player", "Hall", inventory=["Crown"]),
        objective=_objective(ObjectiveType.GET_ITEM, [("Crown", ComponentType.ITEM)]),
    )

    assert world.player.inventory == []
    assert _by_name(world.locations, "Cellar").items == ["Crown"]


def test_objective_item_lying_at_the_start_is_moved_away():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Crown"], connecting=["Cellar"]), _location("Cellar")],
        items=[_item("Crown")],
        characters=[],
        player=_character("Player", "Hall"),
        objective=_objective(ObjectiveType.GET_ITEM, [("Crown", ComponentType.ITEM)]),
    )

    assert _by_name(world.locations, "Hall").items == []
    assert _by_name(world.locations, "Cellar").items == ["Crown"]


def test_item_already_held_by_its_recipient_is_moved_away():
    world, report = _reconcile(
        locations=[_location("Hall", connecting=["Cellar"]), _location("Cellar")],
        items=[_item("Scroll")],
        characters=[_character("Librarian", "Hall", inventory=["Scroll"])],
        player=_character("Player", "Cellar"),
        objective=_objective(ObjectiveType.DELIVER_AN_ITEM,
                             [("Scroll", ComponentType.ITEM), ("Librarian", ComponentType.CHARACTER)]),
    )

    assert _by_name(world.characters, "Librarian").inventory == []
    assert _by_name(world.locations, "Cellar").items == ["Scroll"]


def test_objective_items_and_keys_are_made_gettable():
    world, report = _reconcile(
        locations=[_location("Hall", items=["Brass Key"], connecting=["Vault"], blocked=[_door("Vault", "Brass Key")]),
                   _location("Vault", items=["Crown"])],
        items=[_item("Crown", gettable=False), _item("Brass Key", gettable=False)],
        characters=[],
        player=_character("Player", "Hall"),
        objective=_objective(ObjectiveType.GET_ITEM, [("Crown", ComponentType.ITEM)]),
    )

    assert all(item.gettable for item in world.items)
    assert len([m for m in report.repairs if "made gettable" in m]) == 2


def test_items_given_or_requested_by_characters_are_made_gettable():
    interaction = CharacterInteraction(
        gives_item="Map",
        requires=[ItemRequirement(description="bring it", item_name="Coin")],
        interaction_text="Hello",
        relevance_to_objective="r",
    )
    world, report = _reconcile(
        locations=[_location("Hall", items=["Coin"])],
        items=[_item("Map", gettable=False), _item("Coin", gettable=False)],
        characters=[_character("Mira", "Hall", inventory=["Map"], interaction=interaction)],
        player=_character("Player", "Hall"),
    )

    assert all(item.gettable for item in world.items)


def test_is_objective_target_follows_the_objective():
    world, _ = _reconcile(
        locations=[_location("Hall", items=["Coin"], connecting=["Vault"]), _location("Vault", items=["Crown"])],
        items=[_item("Crown"), GeneratedItem(name="Coin", action_type=ItemActionType.LORE,
                                             descriptions=["d"], is_objective_target=True)],
        characters=[],
        player=_character("Player", "Hall"),
        objective=_objective(ObjectiveType.GET_ITEM, [("Crown", ComponentType.ITEM)]),
    )

    assert {item.name: item.is_objective_target for item in world.items} == {"Crown": True, "Coin": False}


def test_a_mystery_objective_is_its_culprit():
    objective = _mystery([_clue("Ink stain", "Letter")])
    objective.mystery_culprit = "Butler"
    world, report = _reconcile(
        locations=[_location("Hall", items=["Letter"])],
        items=[_item("Letter")],
        characters=[_character("Butler", "Hall")],
        player=_character("Player", "Hall"),
        objective=objective,
    )

    assert [(c.name, c.component_type) for c in world.objective.components] == [("Butler", ComponentType.CHARACTER)]
    assert any("replaced by its culprit" in m for m in report.repairs)
