"""Tests for the facet pipeline wiring, with a fake LLM that returns canned JSON per schema."""

import threading

import pytest

from src.ivie.llm import facet_pipeline, generation_pipeline
from src.ivie.llm.structured_data_models import KeyEntity, WorldConcept, WorldSkeleton

CONCEPT = WorldConcept(title="T", backstory="B", player_concept="P", main_objective="O")
SKELETON = WorldSkeleton(
    key_locations=[KeyEntity(name=name, purpose="p") for name in ["El Jardín", "Torre", "Sótano", "Bóveda"]],
    key_items=[KeyEntity(name=name, purpose="p") for name in ["Llave", "Corona", "Mapa", "Vela"]],
    key_characters=[KeyEntity(name=name, purpose="p") for name in ["Mago", "Guardia"]],
)


def _location(name, connections, items=()):
    return {"name": name, "descriptions": [name], "items": list(items), "connecting_locations": connections,
            "blocked_passages": []}


def _item(name):
    return {"name": name, "action_type": "lore", "descriptions": [name], "gettable": True}


def _character(name, location, inventory=()):
    return {"name": name, "descriptions": [name], "location": location, "inventory": list(inventory)}


LOCATIONS = {"locations": [
    _location("El Jardín", ["Torre"], ["Vela"]),
    _location("Torre", ["El Jardín", "Sótano"]),
    _location("Sótano", ["Torre", "Bóveda"], ["Llave"]),
    _location("Bóveda", ["Sótano"], ["Corona"]),
]}
ITEMS = {"items": [_item(name) for name in ["Llave", "Corona", "Mapa", "Vela"]]}
CHARACTERS = {"characters": [_character("Mago", "Torre", ["Mapa"]), _character("Guardia", "Sótano")],
              "player": _character("Jugador", "El Jardín")}
OBJECTIVE = {"objective": {"type": "get_item", "description": "Get the crown", "success_conditions": ["has it"],
                           "components": [{"name": "Corona", "component_type": "item", "role_in_objective": "goal"}]},
             "world_theme": "t", "narrative_context": "n"}
PUZZLES = {"puzzles": [{"name": "Acertijo", "puzzle_type": "riddle", "descriptions": ["d"], "problem": "p",
                        "answer": "a", "proposed_by_character": "Mago",
                        "rewards": [{"reward_type": "item", "description": "d", "item_name": "Mapa"}],
                        "relevance_to_objective": "r", "hint": "h"}],
           "blocked_passage_additions": [{"from_location": "Sótano", "passage": {
               "location": "Bóveda", "obstacle_name": "Puerta", "obstacle_description": "d",
               "required_to_unblock": {"requirement_type": "puzzle", "description": "d", "puzzle_name": "Acertijo"},
               "relevance_to_objective": "r"}}]}

CANNED = {"WorldLocationsFacet": LOCATIONS, "WorldItemsFacet": ITEMS, "WorldCharactersFacet": CHARACTERS,
          "WorldObjectiveFacet": OBJECTIVE, "PuzzleStageResult": PUZZLES,
          "WorldConcept": CONCEPT.model_dump(), "WorldSkeleton": SKELETON.model_dump()}


class FakeModel:
    """Returns canned JSON per schema; `first_responses` overrides the first answer for a schema."""

    def __init__(self, first_responses=None):
        self.first_responses = dict(first_responses or {})
        self.prompts = {}
        self.lock = threading.Lock()

    def prompt_model_structured(self, prompt, schema, **kwargs):
        name = schema.__name__
        with self.lock:
            self.prompts.setdefault(name, []).append(prompt)
            if name in self.first_responses:
                return self.first_responses.pop(name)
        return CANNED[name]


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(facet_pipeline, "get_llm", lambda name: model)
    return model


def test_parallel_and_sequential_produce_the_same_world(fake_model):
    parallel = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=True)
    sequential = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=False)

    assert parallel.model_dump() == sequential.model_dump()
    assert [location.name for location in parallel.locations] == ["El Jardín", "Torre", "Sótano", "Bóveda"]


def test_a_rejected_facet_is_retried_alone_with_the_reason(monkeypatch):
    bad_locations = {"locations": LOCATIONS["locations"][:3]}
    model = FakeModel(first_responses={"WorldLocationsFacet": bad_locations})
    monkeypatch.setattr(facet_pipeline, "get_llm", lambda name: model)

    world = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=True)

    assert len(world.locations) == 4
    assert len(model.prompts["WorldLocationsFacet"]) == 2
    assert "Missing locations: ['Bóveda']" in model.prompts["WorldLocationsFacet"][1]
    assert len(model.prompts["WorldItemsFacet"]) == 1


def test_a_truncated_response_is_a_retry_not_a_crash(monkeypatch):
    model = FakeModel(first_responses={"WorldItemsFacet": None})
    monkeypatch.setattr(facet_pipeline, "get_llm", lambda name: model)

    world = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=False)

    assert len(world.items) == 4
    assert len(model.prompts["WorldItemsFacet"]) == 2


def test_names_are_rewritten_to_the_roster_spelling(monkeypatch):
    characters = {"characters": [_character("mago", "torre", ["mapa"]), _character("Guardia", "Sotano")],
                  "player": _character("Jugador", "Jardin")}
    model = FakeModel(first_responses={"WorldCharactersFacet": characters})
    monkeypatch.setattr(facet_pipeline, "get_llm", lambda name: model)

    world = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=False)

    mago = next(c for c in world.characters if c.name == "Mago")
    assert (mago.location, mago.inventory) == ("Torre", ["Mapa"])
    assert world.player.location == "El Jardín"


def test_a_facet_that_never_validates_fails_the_generation(monkeypatch):
    class AlwaysBad(FakeModel):
        def prompt_model_structured(self, prompt, schema, **kwargs):
            if schema.__name__ == "WorldObjectiveFacet":
                with self.lock:
                    self.prompts.setdefault("WorldObjectiveFacet", []).append(prompt)
                return {**OBJECTIVE, "objective": {**OBJECTIVE["objective"], "components": []}}
            return super().prompt_model_structured(prompt, schema)

    model = AlwaysBad()
    monkeypatch.setattr(facet_pipeline, "get_llm", lambda name: model)

    with pytest.raises(ValueError, match="'objective' facet"):
        facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=True)
    assert len(model.prompts["WorldObjectiveFacet"]) == 3


def test_puzzle_stage_merges_into_the_world(fake_model):
    world = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=False)

    with_puzzles = facet_pipeline.run_puzzle_stage(world, "es", fake_model, (1, 1))

    assert [p.name for p in with_puzzles.puzzles] == ["Acertijo"]
    sotano = next(l for l in with_puzzles.locations if l.name == "Sótano")
    assert [b.location for b in sotano.blocked_passages] == ["Bóveda"]
    mago = next(c for c in with_puzzles.characters if c.name == "Mago")
    assert mago.interaction.proposes_puzzle == "Acertijo"
    assert "EXACTAMENTE 1 puzzle" in fake_model.prompts["PuzzleStageResult"][0]


def test_full_pipeline_with_the_new_options(monkeypatch, fake_model):
    monkeypatch.setattr(generation_pipeline, "get_llm", lambda name: fake_model)
    monkeypatch.setattr(generation_pipeline, "get_generation_flags",
                        lambda config=None: {"facets": True, "puzzle_stage_only": True, "parallel": True})
    messages = []

    world = generation_pipeline.create_world_incrementally("tema", "es", progress_callback=messages.append)

    assert len(world.locations) == 4 and len(world.puzzles) == 1
    assert [m[:1] for m in messages if m[:1] in "📝🦴🌍🧩"] == ["📝", "🦴", "🌍", "🧩"]
    assert "GeneratedWorld" not in fake_model.prompts  # the old whole-world calls are not made


def _puzzles_rewarding(item_name):
    puzzles = {**PUZZLES, "puzzles": [dict(PUZZLES["puzzles"][0])]}
    puzzles["puzzles"][0]["rewards"] = [{"reward_type": "item", "description": "d", "item_name": item_name}]
    return puzzles


def _world_with_player_holding(item_name, fake_model):
    world = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=False)
    for location in world.locations:
        location.items = [name for name in location.items if name != item_name]
    for character in world.characters:
        character.inventory = [name for name in character.inventory if name != item_name]
    world.player.inventory = [item_name]
    return world


def test_a_reward_the_player_already_has_gets_one_retry(fake_model):
    world = _world_with_player_holding("Vela", fake_model)
    fake_model.first_responses["PuzzleStageResult"] = _puzzles_rewarding("Vela")

    result = facet_pipeline.run_puzzle_stage(world, "es", fake_model, (1, 1))

    prompts = fake_model.prompts["PuzzleStageResult"]
    assert len(prompts) == 2
    assert "already carries from the start" in prompts[1]
    assert result.puzzles[0].rewards[0].item_name == "Mapa"


def test_a_minor_issue_is_accepted_when_the_retry_is_worse(fake_model):
    world = _world_with_player_holding("Vela", fake_model)

    class Sequence:
        def __init__(self):
            self.responses = [_puzzles_rewarding("Vela"), None, None]
            self.prompts = []

        def prompt_model_structured(self, prompt, schema, **kwargs):
            self.prompts.append(prompt)
            return self.responses.pop(0)

    model = Sequence()
    result = facet_pipeline.run_puzzle_stage(world, "es", model, (1, 1))

    assert len(model.prompts) == 2
    assert result.puzzles[0].rewards[0].item_name == "Vela"


def test_an_unknown_objective_component_is_reported_with_the_valid_names():
    from src.ivie.llm.name_matching import NameRoster
    from src.ivie.llm.structured_data_models import WorldObjectiveFacet

    roster = NameRoster(location=[], item=[], character=["Capitán Dubois", "Madame Evangeline"])
    facet = WorldObjectiveFacet.model_validate({**OBJECTIVE, "objective": {
        **OBJECTIVE["objective"], "type": "find_character",
        "components": [{"name": "Dr. Alistair Finch", "component_type": "character", "role_in_objective": "g"}]}})

    errors = facet_pipeline._verify_objective(facet, roster)

    assert "['Capitán Dubois', 'Madame Evangeline']" in errors[0]
