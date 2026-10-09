"""solvability: play a generated world forward to find goals the player can never reach."""

from src.ivie.llm.solvability import ensure_solvable, progress, unmet_goals
from src.ivie.llm.structured_data_models import GeneratedWorld


def _world(*, keeper_at="Interior", objective=None, puzzles=(), keeper_requires=()):
    """Path - Exterior -(key)-> Interior - Study. The Keeper holds the key; the Logbook is in the Study."""
    def location(name, connecting, items=(), blocked=()):
        return {"name": name, "descriptions": ["d"], "items": list(items), "connecting_locations": connecting,
                "blocked_passages": list(blocked)}

    key_lock = {"location": "Interior", "obstacle_name": "Door", "obstacle_description": "d", "relevance_to_objective": "r",
                "required_to_unblock": {"requirement_type": "item", "description": "d", "item_name": "Key"}}
    return GeneratedWorld.model_validate({
        "locations": [location("Path", ["Exterior"]), location("Exterior", ["Path", "Interior"], blocked=[key_lock]),
                      location("Interior", ["Exterior", "Study"]), location("Study", ["Interior"], items=["Logbook"])],
        "items": [{"name": name, "action_type": "lore", "descriptions": ["d"]} for name in ("Key", "Logbook", "Coin")],
        "characters": [{"name": "Keeper", "descriptions": ["d"], "location": keeper_at, "inventory": ["Key"],
                        "interaction": {"gives_item": "Key", "relevance_to_objective": "r",
                                        "requires": list(keeper_requires)}}],
        "puzzles": list(puzzles),
        "player": {"name": "Player", "descriptions": ["d"], "location": "Path"},
        "objective": objective or {"type": "get_item", "description": "Get the logbook", "success_conditions": ["s"],
                                   "components": [{"name": "Logbook", "component_type": "item", "role_in_objective": "g"}]},
        "dependency_chains": [], "world_theme": "t", "narrative_context": "n",
    })


def test_a_key_held_in_front_of_its_door_makes_the_world_solvable():
    world = _world(keeper_at="Exterior")

    assert unmet_goals(world) == []
    assert "Study" in progress(world).reachable


def test_a_key_held_behind_its_own_door_is_reported():
    world = _world(keeper_at="Interior")

    assert unmet_goals(world) == ["the item 'Logbook' can never be obtained"]


def test_the_lock_with_its_key_behind_it_is_opened():
    fixed, report, unmet = ensure_solvable(_world(keeper_at="Interior"))

    assert unmet == []
    assert fixed.locations[1].blocked_passages == []
    assert any("could only be reached behind it" in message for message in report.repairs)


def test_a_solvable_world_is_left_untouched():
    world = _world(keeper_at="Exterior")

    fixed, report, unmet = ensure_solvable(world)

    assert unmet == [] and report.total == 0 and len(fixed.locations[1].blocked_passages) == 1


def test_a_character_withholds_the_reward_of_their_unsolved_puzzle_until_it_is_solvable():
    riddle = {"name": "Riddle", "puzzle_type": "riddle", "descriptions": ["d"], "problem": "¿Qué tiene agujas?",
              "answer": "reloj", "proposed_by_character": "Keeper", "relevance_to_objective": "r", "hint": "h",
              "rewards": [{"reward_type": "item", "description": "d", "item_name": "Key"}]}

    state = progress(_world(keeper_at="Exterior", puzzles=[riddle]))

    assert "Riddle" in state.solved and "Key" in state.obtainable


def test_an_unmet_requirement_keeps_the_item_with_its_character():
    wants_coin = {"requirement_type": "item", "description": "a coin", "item_name": "Coin"}

    state = progress(_world(keeper_at="Exterior", keeper_requires=[wants_coin]))

    assert "Key" not in state.obtainable


def test_a_mystery_clue_only_needs_its_item_in_view():
    mystery = {"type": "solve_mystery", "description": "Who?", "success_conditions": ["s"], "components": [],
               "mystery_culprit": "Keeper", "mystery_question": "Who?", "mystery_solution": "The Keeper",
               "mystery_clues": [{"name": "Ink", "description": "The Logbook has ink", "associated_item": "Logbook",
                                  "relevance_to_mystery": "r"}]}

    assert unmet_goals(_world(keeper_at="Exterior", objective=mystery)) == []
    assert unmet_goals(_world(keeper_at="Interior", objective=mystery)) == ["the clue item 'Logbook' can never be reached"]
