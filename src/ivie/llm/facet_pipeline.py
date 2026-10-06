"""Faster world generation for steps 3 and 4.

Step 3 (facets): instead of one call that writes the whole world, four independent calls write
the locations, items, characters and objective. They all receive the same name roster from the
skeleton, so they can run concurrently, and reconcile_world_facets merges them deterministically.
Each facet is verified and retried on its own, with the rejection reasons fed back into the
prompt, so a bad character list no longer throws away good locations.

Step 4 (puzzle stage): the model still reads the whole world, but returns only the new puzzles
and blocked passages, which merge_puzzle_stage folds into the world. It can no longer move items,
invent elements or break connectivity, and its output is a fraction of the old one.

Both are enabled from the [Generation] section of config.ini; the original single-call steps in
generation_pipeline.py remain available with the flags off.
"""

import contextlib
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import facet_prompts
from .facet_reconciler import reconcile_world_facets, merge_puzzle_stage, _distances_from
from .generation_pipeline import (
    FACET_OUTPUT_TOKENS,
    verify_location_connectivity,
    verify_objective_completability,
    verify_puzzle_rewards_and_fix,
    place_unplaced_required_items,
)
from .models import get_llm
from .name_matching import NameRoster
from .structured_data_models import (
    ItemReward,
    PassageReward,
    PuzzleStageResult,
    WorldCharactersFacet,
    WorldItemsFacet,
    WorldLocationsFacet,
    WorldObjectiveFacet,
)


def _enum_value(value) -> str:
    return (value.value if hasattr(value, 'value') else str(value or '')).lower()


def _coerce(response, model_class):
    """Turn a prompt_model_structured response into `model_class`, or raise.

    prompt_model_structured returns None or {} when the JSON came back truncated or the call
    failed; that is treated as a rejected attempt, not a crash.
    """
    if isinstance(response, model_class):
        return response
    if isinstance(response, dict) and response:
        return model_class.model_validate(response)
    raise ValueError("the model returned no usable output (empty or truncated JSON)")


def _short(error: Exception, limit: int = 600) -> str:
    text = str(error)
    return text if len(text) <= limit else text[:limit] + "..."


#---- Name canonicalization ---------------------------------------------------

def _canonicalize_requirement(requirement, roster: NameRoster) -> None:
    if requirement is None:
        return
    for field, kind in (('item_name', 'item'), ('location_name', 'location'),
                        ('character_name', 'character'), ('puzzle_name', 'puzzle')):
        value = getattr(requirement, field, None)
        if value:
            setattr(requirement, field, roster.canonical(kind, value))


def _canonicalize_locations(facet: WorldLocationsFacet, roster: NameRoster) -> None:
    for location in facet.locations:
        location.name = roster.canonical('location', location.name)
        location.connecting_locations = roster.canonical_list('location', location.connecting_locations)
        location.items = roster.canonical_list('item', location.items)
        for blocked in location.blocked_passages:
            blocked.location = roster.canonical('location', blocked.location)
            _canonicalize_requirement(blocked.required_to_unblock, roster)


def _canonicalize_items(facet: WorldItemsFacet, roster: NameRoster) -> None:
    for item in facet.items:
        item.name = roster.canonical('item', item.name)


def _canonicalize_characters(facet: WorldCharactersFacet, roster: NameRoster) -> None:
    for character in facet.characters:
        character.name = roster.canonical('character', character.name)
    for character in list(facet.characters) + [facet.player]:
        character.location = roster.canonical('location', character.location)
        character.inventory = roster.canonical_list('item', character.inventory)
        interaction = character.interaction
        if interaction is not None:
            interaction.gives_item = roster.canonical('item', interaction.gives_item)
            for requirement in interaction.requires:
                _canonicalize_requirement(requirement, roster)


def _canonicalize_objective(facet: WorldObjectiveFacet, roster: NameRoster) -> None:
    objective = facet.objective
    for component in objective.components:
        component.name = roster.canonical(_enum_value(component.component_type), component.name)
    for clue in objective.mystery_clues or []:
        clue.associated_item = roster.canonical('item', clue.associated_item)
        clue.item_location = roster.canonical('location', clue.item_location)


def _canonicalize_puzzle_stage(stage: PuzzleStageResult, roster: NameRoster) -> None:
    for puzzle in stage.puzzles:
        puzzle.location = roster.canonical('location', puzzle.location)
        # The engine reads proposed_by_location as an item name (see merge_puzzle_stage)
        puzzle.proposed_by_location = roster.canonical('item', puzzle.proposed_by_location)
        puzzle.proposed_by_character = roster.canonical('character', puzzle.proposed_by_character)
        for reward in puzzle.rewards:
            if isinstance(reward, ItemReward):
                reward.item_name = roster.canonical('item', reward.item_name)
            elif isinstance(reward, PassageReward):
                reward.from_location = roster.canonical('location', reward.from_location)
                reward.to_location = roster.canonical('location', reward.to_location)
    for addition in stage.blocked_passage_additions:
        addition.from_location = roster.canonical('location', addition.from_location)
        addition.passage.location = roster.canonical('location', addition.passage.location)
        _canonicalize_requirement(addition.passage.required_to_unblock, roster)


#---- Facet verification --------------------------------------------------------

def _check_roster_names(written: list, expected: list, label: str) -> list:
    """Every roster name exactly once, and nothing else."""
    errors = []
    duplicates = sorted({name for name in written if written.count(name) > 1})
    missing = [name for name in expected if name not in written]
    unknown = [name for name in written if name not in expected]
    if duplicates:
        errors.append(f"Duplicate {label}: {duplicates}. Each one must appear exactly once.")
    if missing:
        errors.append(f"Missing {label}: {missing}. You must develop all of them.")
    if unknown:
        errors.append(f"Unknown {label}: {unknown}. Use only the names from the list, spelled exactly.")
    return errors


def _verify_objective(facet: WorldObjectiveFacet, roster: NameRoster) -> list:
    errors = []
    objective = facet.objective
    objective_type = _enum_value(objective.type)

    components = {}
    for component in objective.components:
        kind = _enum_value(component.component_type)
        if component.name not in roster.names(kind):
            # List the valid names: the concept often centres on someone the skeleton left out
            # (a kidnapped scientist, say), and without the list the model repeats the same name
            errors.append(f"Objective component '{component.name}' is not a {kind} from the list. "
                          f"Valid {kind} names: {sorted(roster.names(kind))}. If the goal involves something "
                          f"that is not on the lists, build the objective around an element that is, "
                          f"or choose another objective type.")
        components.setdefault(kind, []).append(component.name)

    if objective_type == 'solve_mystery':
        clues = objective.mystery_clues or []
        if not clues:
            errors.append("A solve_mystery objective needs mystery_clues.")
        clue_items = [clue.associated_item for clue in clues]
        unknown = [name for name in clue_items if name not in roster.names('item')]
        if unknown:
            errors.append(f"Mystery clues name items that are not in the item list: {unknown}. "
                          f"Valid item names: {sorted(roster.names('item'))}.")
        duplicates = sorted({name for name in clue_items if clue_items.count(name) > 1})
        if duplicates:
            errors.append(f"Several clues share the same associated_item {duplicates}; "
                          f"each clue must use a different item.")
        if not objective.mystery_solution:
            errors.append("A solve_mystery objective needs a mystery_solution.")
    elif objective_type == 'deliver_an_item':
        if not components.get('item'):
            errors.append("A deliver_an_item objective needs one component of type 'item' (what is delivered).")
        if not components.get('character') and not components.get('location'):
            errors.append("A deliver_an_item objective needs one component of type 'character' or 'location' "
                          "(the recipient).")
    else:
        needed = {'reach_location': 'location', 'get_item': 'item', 'find_character': 'character'}.get(objective_type)
        if needed and not components.get(needed):
            errors.append(f"A {objective_type} objective needs one component of type '{needed}'.")
    return errors


#---- Running facets -----------------------------------------------------------

def _run_facet(model_name, label, build_prompt, facet_class, canonicalize, verify,
               max_attempts=3, initial_errors=None):
    """Generate one facet, retrying it alone with the rejection reasons fed back.

    Runs in a worker thread when facets are parallel: it must not touch Streamlit, and it uses
    its own LLM client.
    """
    model = get_llm(model_name)
    previous_errors = initial_errors
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = model.prompt_model_structured(build_prompt(previous_errors), facet_class,
                                                     max_output_tokens=FACET_OUTPUT_TOKENS)
            facet = _coerce(response, facet_class)
            canonicalize(facet)
            errors = verify(facet)
        except Exception as error:
            facet, errors = None, [f"Invalid output: {_short(error)}"]
        print(f"[TIMING] faceta '{label}' (intento {attempt}): {time.perf_counter() - started:.1f}s")
        if not errors:
            return facet
        print(f"[WARNING] Facet '{label}' rejected on attempt {attempt}: {errors}")
        previous_errors = errors
    raise ValueError(f"Failed to generate a valid '{label}' facet after {max_attempts} attempts: {previous_errors}")


def _run_wave(jobs: dict, parallel: bool) -> dict:
    """Run {label: callable} jobs, concurrently or one after another."""
    results = {}
    if parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {executor.submit(job): label for label, job in jobs.items()}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    else:
        for label, job in jobs.items():
            results[label] = job()
    return results


def _verify_assembled_world(world) -> dict:
    """Checks on the reconciled world, as {facet to regenerate: reasons}."""
    errors = {}
    if not verify_location_connectivity(world):
        reachable = _distances_from(world.player.location, world.locations)
        unreachable = [location.name for location in world.locations if location.name not in reachable]
        errors['locations'] = [f"These locations are not connected to the rest of the world: {unreachable}. "
                               f"Every location must be reachable from every other one."]

    # verify_objective_completability reports its reasons by printing them; capture them as feedback
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        objective_ok = verify_objective_completability(world)
    print(output.getvalue(), end="")
    if not objective_ok:
        reasons = [line.replace("[ERROR]", "").strip() for line in output.getvalue().splitlines() if "[ERROR]" in line]
        errors['objective'] = reasons or ["The objective cannot be completed with the elements of the world."]
    return errors


def generate_world_facets(concept, skeleton, language, model_name, parallel=True,
                          max_attempts=3, max_rounds=2):
    """Step 3: generate the world as four facets and reconcile them into a GeneratedWorld."""
    expected = {
        'location': [entity.name for entity in skeleton.key_locations],
        'item': [entity.name for entity in skeleton.key_items],
        'character': [entity.name for entity in skeleton.key_characters],
    }
    roster = NameRoster(**expected)

    specs = {
        'locations': (facet_prompts.PROMPT_FACET_LOCATIONS, WorldLocationsFacet, _canonicalize_locations,
                      lambda facet: _check_roster_names([l.name for l in facet.locations], expected['location'], "locations")),
        'items': (facet_prompts.PROMPT_FACET_ITEMS, WorldItemsFacet, _canonicalize_items,
                  lambda facet: _check_roster_names([i.name for i in facet.items], expected['item'], "items")),
        'characters': (facet_prompts.PROMPT_FACET_CHARACTERS, WorldCharactersFacet, _canonicalize_characters,
                       lambda facet: _check_roster_names([c.name for c in facet.characters], expected['character'], "characters")),
        'objective': (facet_prompts.PROMPT_FACET_OBJECTIVE, WorldObjectiveFacet, _canonicalize_objective,
                      lambda facet: _verify_objective(facet, roster)),
    }

    def job(label, initial_errors=None):
        prompt_builder, facet_class, canonicalize, verify = specs[label]
        return lambda: _run_facet(
            model_name, label,
            lambda previous_errors: prompt_builder(concept, skeleton, language, previous_errors=previous_errors),
            facet_class,
            lambda facet: canonicalize(facet, roster),
            verify,
            max_attempts=max_attempts,
            initial_errors=initial_errors,
        )

    facets = _run_wave({label: job(label) for label in specs}, parallel)

    for round_number in range(1, max_rounds + 1):
        roster.log_rewrites()
        world, report = reconcile_world_facets(
            facets['locations'], facets['items'], facets['characters'], facets['objective'])
        report.log()
        errors = _verify_assembled_world(world)
        if not errors:
            return world
        print(f"[WARNING] Reconciled world rejected (round {round_number}): {errors}")
        if round_number < max_rounds:
            facets.update(_run_wave({label: job(label, reasons) for label, reasons in errors.items()}, parallel))

    raise ValueError(f"Failed to assemble a valid world from facets: {errors}")


#---- Puzzle stage --------------------------------------------------------------

def _valid_names_hint(world) -> str:
    return (f"Valid names - locations: {[l.name for l in world.locations]}; "
            f"items: {[i.name for i in world.items]}; characters: {[c.name for c in world.characters]}.")


def _pointless_challenges(merged, original_world) -> list:
    """Puzzles and blocks the player gets past for free: rewards or keys already in their hands."""
    carried = set(original_world.player.inventory)
    problems = []
    for puzzle in merged.puzzles:
        for reward in puzzle.rewards:
            if isinstance(reward, ItemReward) and reward.item_name in carried:
                problems.append(f"Puzzle '{puzzle.name}' rewards '{reward.item_name}', which the player already "
                                f"carries from the start. Reward something the player does not have yet.")
    existing_blocks = {(l.name, b.location) for l in original_world.locations for b in l.blocked_passages}
    for location in merged.locations:
        for blocked in location.blocked_passages:
            item_name = getattr(blocked.required_to_unblock, 'item_name', None)
            if (location.name, blocked.location) not in existing_blocks and item_name in carried:
                problems.append(f"The new block '{location.name}' -> '{blocked.location}' opens with '{item_name}', "
                                f"which the player already carries, so it would open immediately.")
    return problems


def run_puzzle_stage(world, language, model, puzzle_range, max_attempts=3):
    """Step 4: add puzzles and blocked passages to a finished world without rewriting it.

    Hard errors (no valid puzzle, broken connectivity or objective) are retried and finally fail
    the generation. Soft ones (a puzzle or block the player gets past for free) get one retry;
    the first acceptable result is kept, since a pointless puzzle is not worth failing a world over.
    """
    min_puzzles, max_puzzles = puzzle_range
    puzzle_count = str(min_puzzles) if min_puzzles == max_puzzles else f"{min_puzzles}-{max_puzzles}"
    world_json = world.model_dump_json(indent=2)

    previous_errors = None
    acceptable = None
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        errors, soft_errors = [], []
        merged = None
        try:
            prompt = facet_prompts.PROMPT_PUZZLE_STAGE(world_json, puzzle_count, language, previous_errors=previous_errors)
            response = model.prompt_model_structured(prompt, PuzzleStageResult, max_output_tokens=FACET_OUTPUT_TOKENS)
            stage = _coerce(response, PuzzleStageResult)

            roster = NameRoster(
                location=[location.name for location in world.locations],
                item=[item.name for item in world.items],
                character=[character.name for character in world.characters],
                puzzle=[puzzle.name for puzzle in stage.puzzles],
            )
            _canonicalize_puzzle_stage(stage, roster)
            roster.log_rewrites()

            merged, report = merge_puzzle_stage(world, stage, max_puzzles=max_puzzles)
            report.log("[PUZZLES]")
            if len(merged.puzzles) < min_puzzles:
                errors.append(f"Only {len(merged.puzzles)} valid puzzle(s); create exactly {puzzle_count}.")
                errors.extend(report.repairs)
                errors.append(_valid_names_hint(world))

            _, merged = verify_puzzle_rewards_and_fix(merged)
            merged = place_unplaced_required_items(merged)
            if not verify_location_connectivity(merged):
                errors.append("The blocked passages left some locations unreachable.")
            if not verify_objective_completability(merged):
                errors.append("After adding the puzzles, the objective can no longer be completed.")
            soft_errors = _pointless_challenges(merged, world)
        except Exception as error:
            errors.append(f"Invalid output: {_short(error)}")

        print(f"[TIMING] etapa de puzzles (intento {attempt}): {time.perf_counter() - started:.1f}s")
        if not errors and not soft_errors:
            return merged
        if not errors:
            if acceptable is not None:
                print(f"[WARNING] Puzzle stage accepted with minor issues: {soft_errors}")
                return merged
            acceptable = merged
        elif acceptable is not None:
            print("[WARNING] Puzzle stage retry failed; keeping the earlier acceptable result")
            return acceptable
        print(f"[WARNING] Puzzle stage rejected on attempt {attempt}: {errors + soft_errors}")
        previous_errors = errors + soft_errors

    if acceptable is not None:
        return acceptable
    raise ValueError(f"Failed to generate valid puzzles after {max_attempts} attempts: {previous_errors}")
