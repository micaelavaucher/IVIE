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
PUZZLES = {"puzzles": [{"name": "Acertijo", "puzzle_type": "riddle", "descriptions": ["d"], "problem": "¿Qué tiene agujas pero no cose?",
                        "answer": "reloj", "proposed_by_character": "Mago",
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


def _objective_facet(objective_type, components):
    from src.ivie.llm.structured_data_models import WorldObjectiveFacet
    return WorldObjectiveFacet.model_validate({**OBJECTIVE, "objective": {
        **OBJECTIVE["objective"], "type": objective_type,
        "components": [{"name": n, "component_type": t, "role_in_objective": "g"} for n, t in components]}})


def test_a_component_repeated_five_times_is_rejected():
    from src.ivie.llm.name_matching import NameRoster
    roster = NameRoster(location=[], item=["Fragmento (1 de 5)"], character=[])
    facet = _objective_facet("get_item", [("Fragmento (1 de 5)", "item")] * 5)

    errors = facet_pipeline._verify_objective(facet, roster)

    assert any("repeats components" in e for e in errors)


def test_deliver_takes_one_item_and_one_recipient():
    from src.ivie.llm.name_matching import NameRoster
    roster = NameRoster(location=["Torre"], item=["Corona"], character=["Mago"])
    ok = _objective_facet("deliver_an_item", [("Corona", "item"), ("Mago", "character")])
    too_many = _objective_facet("deliver_an_item", [("Corona", "item"), ("Mago", "character"), ("Torre", "location")])

    assert facet_pipeline._verify_objective(ok, roster) == []
    assert facet_pipeline._verify_objective(too_many, roster)


def test_making_a_listed_npc_the_player_is_explained():
    from src.ivie.llm.structured_data_models import WorldCharactersFacet
    facet = WorldCharactersFacet.model_validate({
        "characters": [_character("Sherlock Holmes", "Torre")],
        "player": _character("Dr. John Watson", "Torre")})

    errors = facet_pipeline._verify_characters(facet, ["Sherlock Holmes", "Dr. John Watson"])

    assert any("you made them the player" in e for e in errors)


def test_a_puzzle_nobody_proposes_is_flagged(fake_model):
    world = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=False)
    unreachable = {**PUZZLES, "puzzles": [dict(PUZZLES["puzzles"][0], proposed_by_character=None, name="Huérfano")],
                   "blocked_passage_additions": []}
    fake_model.first_responses["PuzzleStageResult"] = unreachable

    facet_pipeline.run_puzzle_stage(world, "es", fake_model, (1, 1))

    assert "nobody presents it" in fake_model.prompts["PuzzleStageResult"][1]


def test_a_failed_assembly_gets_one_new_skeleton(monkeypatch, fake_model):
    monkeypatch.setattr(generation_pipeline, "get_llm", lambda name: fake_model)
    monkeypatch.setattr(generation_pipeline, "get_generation_flags",
                        lambda config=None: {"facets": True, "puzzle_stage_only": True, "parallel": False})
    real = facet_pipeline.generate_world_facets
    calls = []

    def fails_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("Failed to generate a valid 'objective' facet")
        return real(*args, **kwargs)

    monkeypatch.setattr(facet_pipeline, "generate_world_facets", fails_once)

    world = generation_pipeline.create_world_incrementally("tema", "es")

    assert len(calls) == 2 and len(fake_model.prompts["WorldSkeleton"]) == 2
    assert len(world.locations) == 4


def test_a_skeleton_name_with_a_newline_is_rejected():
    broken = SKELETON.model_copy(deep=True)
    broken.key_locations[0].name = "Anticuario 'El Cron\nmetro Roto'"

    assert generation_pipeline.skeleton_names_are_unique(SKELETON)
    assert not generation_pipeline.skeleton_names_are_unique(broken)


def test_gathering_several_different_items_is_a_valid_objective():
    from src.ivie.llm.name_matching import NameRoster
    roster = NameRoster(location=[], item=["Fragmento Norte", "Fragmento Sur", "Fragmento Este"], character=[])
    facet = _objective_facet("get_item", [("Fragmento Norte", "item"), ("Fragmento Sur", "item"), ("Fragmento Este", "item")])

    assert facet_pipeline._verify_objective(facet, roster) == []


def _stage_with(**puzzle_fields):
    puzzle = dict(PUZZLES["puzzles"][0], **puzzle_fields)
    return {**PUZZLES, "puzzles": [puzzle], "blocked_passage_additions": []}


def test_a_statement_that_is_only_an_instruction_is_unplayable(fake_model):
    world = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=False)
    fake_model.first_responses["PuzzleStageResult"] = _stage_with(
        problem="Resuelve el acertijo del Mago para ganar su confianza")

    result = facet_pipeline.run_puzzle_stage(world, "es", fake_model, (1, 1))

    assert "does not state the actual question" in fake_model.prompts["PuzzleStageResult"][1]
    assert result.puzzles[0].problem == "¿Qué tiene agujas pero no cose?"


def test_a_puzzle_still_unplayable_after_the_retry_is_dropped(monkeypatch):
    bad = _stage_with(problem="Solve the riddle")

    class AlwaysBad(FakeModel):
        def prompt_model_structured(self, prompt, schema, **kwargs):
            if schema.__name__ == "PuzzleStageResult":
                self.prompts.setdefault("PuzzleStageResult", []).append(prompt)
                return bad
            return super().prompt_model_structured(prompt, schema, **kwargs)

    model = AlwaysBad()
    monkeypatch.setattr(facet_pipeline, "get_llm", lambda name: model)
    world = facet_pipeline.generate_world_facets(CONCEPT, SKELETON, "es", "fake", parallel=False)

    result = facet_pipeline.run_puzzle_stage(world, "es", model, (1, 1))

    assert len(model.prompts["PuzzleStageResult"]) == 2
    assert result.puzzles == []
    assert all(c.interaction is None or c.interaction.proposes_puzzle is None for c in result.characters)


def test_a_statement_with_its_own_answer_is_unplayable():
    from src.ivie.llm.structured_data_models import GeneratedPuzzle
    puzzle = GeneratedPuzzle(name="R", puzzle_type="riddle", descriptions=["d"], problem="¿Qué abre esta puerta? La llave dorada",
                             answer="la llave dorada", proposed_by_character="Mago", rewards=[], relevance_to_objective="r", hint="h")

    class World:
        locations = []
        characters = []
        puzzles = [puzzle]

    assert "contains its own answer" in facet_pipeline.unplayable_puzzles(World())["R"]


#---- Mysteries: who is responsible ---------------------------------------------------

def _mystery_facet(**overrides):
    from src.ivie.llm.structured_data_models import WorldObjectiveFacet
    objective = {
        "type": "solve_mystery", "description": "Descubre quién robó la corona", "success_conditions": ["acusa"],
        "components": [{"name": "Guardia", "component_type": "character", "role_in_objective": "culpable"}],
        "mystery_question": "¿Quién robó la corona?", "mystery_culprit": "Guardia",
        "mystery_solution": "El Guardia robó la corona durante su ronda",
        "mystery_clues": [{"name": "Cera", "description": "La Vela tiene cera fresca del Sótano",
                           "associated_item": "Vela", "relevance_to_mystery": "r"}],
    }
    objective.update(overrides)
    return WorldObjectiveFacet.model_validate({**OBJECTIVE, "objective": objective})


def _mystery_roster():
    from src.ivie.llm.name_matching import NameRoster
    return NameRoster(location=["Torre"], item=["Vela", "Llave"], character=["Mago", "Guardia"])


def test_a_well_formed_mystery_passes():
    assert facet_pipeline._verify_objective(_mystery_facet(), _mystery_roster()) == []


def test_a_mystery_needs_a_culprit_among_the_characters():
    missing = facet_pipeline._verify_objective(_mystery_facet(mystery_culprit=None), _mystery_roster())
    outsider = facet_pipeline._verify_objective(_mystery_facet(mystery_culprit="Silas el Sombrío"), _mystery_roster())

    assert any("mystery_culprit" in e and "get_item" in e for e in missing)
    assert any("mystery_culprit" in e for e in outsider)


def test_the_solution_must_name_the_culprit():
    errors = facet_pipeline._verify_objective(
        _mystery_facet(mystery_solution="Un agente de una sociedad secreta robó la corona"), _mystery_roster())

    assert any("does not name the culprit" in e for e in errors)


def test_a_clue_about_another_object_is_rejected():
    clue = {"name": "Caja de fósforos", "description": "Una caja de fósforos de un club de París",
            "associated_item": "Llave", "relevance_to_mystery": "r"}

    errors = facet_pipeline._verify_objective(_mystery_facet(mystery_clues=[clue]), _mystery_roster())

    assert any("does not mention its item 'Llave'" in e for e in errors)


def test_mentions_accepts_a_partial_reference_to_a_long_name():
    assert facet_pipeline.mentions("Con la lupa ves un hilo rojo", "Lupa de Detective")
    assert not facet_pipeline.mentions("Una caja de fósforos", "A Compartment Key")


def test_a_puzzle_that_changes_nothing_is_unplayable():
    from src.ivie.llm.structured_data_models import GeneratedPuzzle
    puzzle = GeneratedPuzzle(name="R", puzzle_type="riddle", descriptions=["d"], problem="¿Qué tiene agujas pero no cose?",
                             answer="reloj", proposed_by_character="Mago", rewards=[], relevance_to_objective="r", hint="h")

    class World:
        locations = []
        characters = []
        puzzles = [puzzle]

    assert "changes nothing" in facet_pipeline.unplayable_puzzles(World())["R"]


#---- The objective's text must not promise more than its components ----------------------

def test_a_promised_quantity_is_found_only_next_to_an_item_name():
    items = ["Fragmento del Orbe de Armonía", "Mapa"]

    assert facet_pipeline.promised_item_count("Encuentra los tres fragmentos del Orbe", items) == 3
    assert facet_pipeline.promised_item_count("Reassemble the 4 scattered fragments of the Orbe", items) == 4
    assert facet_pipeline.promised_item_count("Viaja entre dos mundos para encontrar el Mapa", items) == 0


def test_a_description_promising_more_items_than_its_components_is_rejected():
    from src.ivie.llm.name_matching import NameRoster
    roster = NameRoster(location=[], item=["Fragmento del Orbe"], character=["Lyra"])
    facet = _objective_facet("deliver_an_item", [("Fragmento del Orbe", "item"), ("Lyra", "character")])
    facet.objective.description = "Encuentra los tres fragmentos del Orbe y entrégaselos a Lyra"

    errors = facet_pipeline._verify_objective(facet, roster)

    assert any("promises 3 items" in e for e in errors)


def test_a_failed_generation_starts_over_from_a_new_concept(monkeypatch, fake_model):
    monkeypatch.setattr(generation_pipeline, "get_llm", lambda name: fake_model)
    monkeypatch.setattr(generation_pipeline, "get_generation_flags",
                        lambda config=None: {"facets": True, "puzzle_stage_only": True, "parallel": False})
    real = generation_pipeline._generate_skeleton
    calls = []

    def skeleton_fails_first(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("Failed to generate a valid skeleton after multiple attempts.")
        return real(*args, **kwargs)

    monkeypatch.setattr(generation_pipeline, "_generate_skeleton", skeleton_fails_first)
    messages = []

    world = generation_pipeline.create_world_incrementally("tema", "es", progress_callback=messages.append)

    assert len(world.locations) == 4 and len(fake_model.prompts["WorldConcept"]) == 2
    assert any(m.startswith("[WARNING] Generation attempt 1 failed") for m in messages)


def test_a_quota_error_is_not_retried(monkeypatch, fake_model):
    monkeypatch.setattr(generation_pipeline, "get_llm", lambda name: fake_model)

    def quota(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(generation_pipeline, "_generate_skeleton", quota)

    with pytest.raises(RuntimeError):
        generation_pipeline.create_world_incrementally("tema", "es")
    assert len(fake_model.prompts["WorldConcept"]) == 1
