"""Incremental world generation pipeline for IVIE.

This module implements a step-by-step world generation system,
dividing the process into logical and sequential phases to improve
the coherence and quality of the final world.
"""

#---- Imports -----------------------------------------------------------------
from .structured_data_models import (
    WorldConcept, 
    WorldSkeleton, 
    GeneratedWorld
)
from .models import get_llm
from . import prompts
from ..config import load_config, get_model_names
import configparser
import time

# Output caps for the small structured calls. A normal answer is far below them; the cap matters
# when the model falls into a repetition loop, which otherwise runs to 8192 tokens (~30s) before
# the truncated JSON is retried.
SMALL_OUTPUT_TOKENS = 2048
FACET_OUTPUT_TOKENS = 4096

#---- Pipeline Functions -----------------------------------------------------

def run_step_1_concept(theme: str, language, model=None) -> WorldConcept:
    if model is None:
        config = load_config()
        reasoning_model_name, _ = get_model_names(config)
        model = get_llm(reasoning_model_name)
    prompt_text = prompts.PROMPT_STEP_1_CONCEPT_BY_THEME(theme=theme, language=language)
    print(f"[DEBUG] Generating world concept with theme: '{theme}'")
    concept_response = model.prompt_model_structured(prompt_text, WorldConcept, max_output_tokens=SMALL_OUTPUT_TOKENS)

    if isinstance(concept_response, dict):
        concept = WorldConcept(**concept_response)
    else:
        concept = concept_response

    print(f"[DEBUG] ✅ Step 1 - Concept generated:")
    print(f"  Title: {concept.title}")
    print(f"  Backstory: {concept.backstory[:100]}...")
    print(f"  Player concept: {concept.player_concept}")
    print(f"  Main objective: {concept.main_objective}")
    return concept

def run_step_1_generate_concept(language, model=None) -> WorldConcept:
    if model is None:
        config = load_config()
        reasoning_model_name, _ = get_model_names(config)
        model = get_llm(reasoning_model_name)
    prompt_text = prompts.PROMPT_STEP_1_CONCEPT(language=language)
    print(f"[DEBUG] Generating a new world concept")
    concept_response = model.prompt_model_structured(prompt_text, WorldConcept, max_output_tokens=SMALL_OUTPUT_TOKENS)

    if isinstance(concept_response, dict):
        concept = WorldConcept(**concept_response)
    else:
        concept = concept_response

    print(f"[DEBUG] ✅ Step 1 - Concept generated:")
    print(f"  Title: {concept.title}")
    print(f"  Backstory: {concept.backstory[:100]}...")
    print(f"  Player concept: {concept.player_concept}")
    print(f"  Main objective: {concept.main_objective}")
    return concept

def run_step_2_skeleton(concept: WorldConcept, language, model=None) -> WorldSkeleton:
    if model is None:
        config = load_config()
        reasoning_model_name, _ = get_model_names(config)
        model = get_llm(reasoning_model_name)
    prompt_text = prompts.PROMPT_STEP_2_SKELETON(
        title=concept.title,
        backstory=concept.backstory,
        player_concept=concept.player_concept,
        main_objective=concept.main_objective,
        language=language
    )
    
    print(f"[DEBUG] Generating skeleton from concept: {concept.title}")
    skeleton_response = model.prompt_model_structured(prompt_text, WorldSkeleton, max_output_tokens=SMALL_OUTPUT_TOKENS)
    
    if isinstance(skeleton_response, dict):
        skeleton = WorldSkeleton(**skeleton_response)
    else:
        skeleton = skeleton_response
        
    print(f"[DEBUG] ✅ Step 2 - Skeleton generated:")
    print(f"  Key locations: {[loc.name for loc in skeleton.key_locations]}")
    print(f"  Key items: {[item.name for item in skeleton.key_items]}")
    print(f"  Key characters: {[char.name for char in skeleton.key_characters]}")
    return skeleton

def run_step_3_details(concept: WorldConcept, skeleton: WorldSkeleton, language, model=None) -> GeneratedWorld:
    if concept is None:
        raise ValueError("Concept parameter cannot be None. Failed to generate world concept in previous step.")
    
    required_attrs = ['title', 'backstory', 'player_concept', 'main_objective']
    for attr in required_attrs:
        if not hasattr(concept, attr) or getattr(concept, attr) is None:
            raise ValueError(f"Concept is missing required attribute '{attr}' or it is None. Failed to generate complete world concept.")
    
    if model is None:
        config = load_config()
        reasoning_model_name, _ = get_model_names(config)
        model = get_llm(reasoning_model_name)

    skeleton_data = f"""
    Ubicaciones clave:
    {chr(10).join([f"- {loc.name}: {loc.purpose}" for loc in skeleton.key_locations])}

    Objetos clave:
    {chr(10).join([f"- {item.name}: {item.purpose}" for item in skeleton.key_items])}

    Personajes clave:
    {chr(10).join([f"- {char.name}: {char.purpose}" for char in skeleton.key_characters])}
    """
    
    prompt_text = prompts.PROMPT_STEP_3_DETAILS(
        title=concept.title,
        backstory=concept.backstory,
        player_concept=concept.player_concept,
        main_objective=concept.main_objective,
        skeleton_data=skeleton_data,
        language=language
    )
    
    print(f"[DEBUG] Generating details from skeleton with {len(skeleton.key_locations)} locations")
    world_response = model.prompt_model_structured(prompt_text, GeneratedWorld)
    
    if isinstance(world_response, dict):
        world = GeneratedWorld(**world_response)
    else:
        world = world_response
        
    print(f"[DEBUG] ✅ Step 3 - Details generated:")
    print(f"  Total locations: {len(world.locations)}")
    print(f"  Location names: {[loc for loc in world.locations]}")
    print(f"  Total items: {len(world.items)}")
    print(f"  Item names: {[item.name for item in world.items]}")
    print(f"  Total characters: {len(world.characters)}")
    print(f"  Character names: {[char.name for char in world.characters]}")
    print(f"  Total puzzles: {len(world.puzzles)}")
    print(f"  Puzzle names: {[puzzle.name for puzzle in world.puzzles]}")
    if world.objective:
        print(f"  Main objective: {world.objective}")
    return world

def run_step_4_puzzles(world_data: GeneratedWorld, language, model=None) -> GeneratedWorld:
    if model is None:
        config = load_config()
        reasoning_model_name, _ = get_model_names(config)
        model = get_llm(reasoning_model_name)
    
    world_json = world_data.model_dump_json(indent=2)
    
    prompt_text = prompts.PROMPT_STEP_4_PUZZLES(
        world_data=world_json,
        language=language
    )
    
    print(f"[DEBUG] Adding puzzles and obstacles to world with {len(world_data.puzzles)} existing puzzles")
    enhanced_world_response = model.prompt_model_structured(prompt_text, GeneratedWorld)
    
    if isinstance(enhanced_world_response, dict):
        enhanced_world = GeneratedWorld(**enhanced_world_response)
    else:
        enhanced_world = enhanced_world_response
        
    print(f"[DEBUG] ✅ Step 4 - Puzzles enhanced:")
    print(f"  Total locations: {len(enhanced_world.locations)}")
    print(f"  Total items: {len(enhanced_world.items)}")
    print(f"  Total characters: {len(enhanced_world.characters)}")
    print(f"  Total puzzles: {len(enhanced_world.puzzles)} (was {len(world_data.puzzles)})")
    print(f"  Puzzle names: {[puzzle.name for puzzle in enhanced_world.puzzles]}")
    for puzzle in enhanced_world.puzzles:
        print(f"  • {puzzle}:")
        if hasattr(puzzle, 'puzzle_hints') and puzzle.puzzle_hints:
            print(f"    {puzzle.name} hints: {len(puzzle.puzzle_hints)} hints:")
            print(f"      " + "\n      ".join([f"- {hint}" for hint in puzzle.puzzle_hints]))
    return enhanced_world

def _report(progress_callback, message: str) -> None:
    print(message)
    if progress_callback:
        progress_callback(message)

def _log_timing(label: str, since: float) -> None:
    print(f"[TIMING] {label}: {time.perf_counter() - since:.1f}s")

def _parse_range(value):
    if '-' in value:
        return tuple(map(int, value.split('-')))
    return int(value), int(value)

def get_generation_flags(config=None) -> dict:
    """Options of the faster pipeline ([Generation] in config.ini). Missing options keep the original pipeline."""
    config = config or load_config()
    return {
        'facets': config.getboolean('Generation', 'Facets', fallback=False),
        'puzzle_stage_only': config.getboolean('Generation', 'PuzzleStageOnly', fallback=False),
        'parallel': config.getboolean('Generation', 'Parallel', fallback=False),
    }

def _get_generation_model():
    config = load_config()
    reasoning_model_name, _ = get_model_names(config)
    model = get_llm(reasoning_model_name)
    print(f"🤖 Using model for world generation: {reasoning_model_name}")
    return model, reasoning_model_name

def create_world_incrementally(theme: str, language: str, progress_callback=None) -> GeneratedWorld:
    print(f"⚙️ Iniciando generación incremental del mundo con tema: '{theme}'")
    started = time.perf_counter()
    model, model_name = _get_generation_model()

    # Paso 1: Generar el concepto general
    _report(progress_callback, "📝 Paso 1: Generando concepto del mundo...")
    step_started = time.perf_counter()
    concept = run_step_1_concept(theme, language, model)
    _log_timing("Paso 1 (concepto)", step_started)
    _report(progress_callback, f"✅ Concepto creado: '{concept.title}'")

    return _create_world_from_concept(concept, language, model, model_name, progress_callback, started)

def create_world_incrementally_generate(language: str, progress_callback=None) -> GeneratedWorld:
    print(f"⚙️ Iniciando generación incremental del mundo en modo 'generate'")
    started = time.perf_counter()
    model, model_name = _get_generation_model()

    # Paso 1: Generar el concepto general
    _report(progress_callback, "📝 Paso 1: Generando concepto del mundo...")
    step_started = time.perf_counter()
    concept = run_step_1_generate_concept(language, model)
    _log_timing("Paso 1 (concepto)", step_started)
    _report(progress_callback, f"✅ Concepto creado: '{concept.title}'")

    return _create_world_from_concept(concept, language, model, model_name, progress_callback, started)

def _create_world_from_concept(concept, language, model, model_name, progress_callback, started) -> GeneratedWorld:
    """Steps 2-4, shared by Inspiration and Generate modes."""
    flags = get_generation_flags()
    print(f"⚙️ Generation options: {flags}")
    max_attempts = 3

    # Paso 2: Crear el esqueleto con entidades clave (with verification)
    _report(progress_callback, "🦴 Paso 2: Creando esqueleto del mundo...")
    step_started = time.perf_counter()
    skeleton = None
    for attempt in range(1, max_attempts + 1):
        skeleton = run_step_2_skeleton(concept, language, model)
        # Verify skeleton sizes
        valid_locations = hasattr(skeleton, 'key_locations') and skeleton.key_locations and isinstance(skeleton.key_locations, list)
        valid_items = hasattr(skeleton, 'key_items') and skeleton.key_items and isinstance(skeleton.key_items, list)
        valid_characters = hasattr(skeleton, 'key_characters') and skeleton.key_characters and isinstance(skeleton.key_characters, list)
        config = configparser.ConfigParser()
        config.read('config.ini')
        locations_range = _parse_range(config['Size_World']['Locations'])
        objects_range = _parse_range(config['Size_World']['Objects'])
        npcs_range = _parse_range(config['Size_World']['NPCs'])
        size_ok = (
            valid_locations and locations_range[0] <= len(skeleton.key_locations) <= locations_range[1] and
            valid_items and objects_range[0] <= len(skeleton.key_items) <= objects_range[1] and
            valid_characters and npcs_range[0] <= len(skeleton.key_characters) <= npcs_range[1]
        )
        json_ok = verify_pydantic_model(skeleton, WorldSkeleton)
        names_ok = size_ok and skeleton_names_are_unique(skeleton)
        if size_ok and json_ok and names_ok:
            break
        else:
            _report(progress_callback, f"[WARNING] Skeleton verification failed on attempt {attempt}. Retrying...")
            skeleton = None
    if skeleton is None:
        raise ValueError("Failed to generate a valid skeleton after multiple attempts.")
    _log_timing("Paso 2 (esqueleto)", step_started)
    _report(progress_callback, f"✅ Esqueleto creado con {len(skeleton.key_locations)} ubicaciones, {len(skeleton.key_items)} objetos y {len(skeleton.key_characters)} personajes")

    # Paso 3: Desarrollar detalles y conexiones (with verification)
    _report(progress_callback, "🌍 Paso 3: Desarrollando detalles y conexiones...")
    step_started = time.perf_counter()
    if flags['facets']:
        from .facet_pipeline import generate_world_facets
        world_basic = generate_world_facets(concept, skeleton, language, model_name, parallel=flags['parallel'])
    else:
        world_basic = None
        for attempt in range(1, max_attempts + 1):
            world_basic = run_step_3_details(concept, skeleton, language, model)
            json_ok = verify_pydantic_model(world_basic, GeneratedWorld)
            connectivity_ok = verify_location_connectivity(world_basic)
            objective_ok = verify_objective_completability(world_basic)
            if json_ok and connectivity_ok and objective_ok:
                break
            else:
                if not json_ok:
                    _report(progress_callback, f"[WARNING] Details JSON verification failed on attempt {attempt}. Retrying...")
                if not connectivity_ok:
                    _report(progress_callback, f"[WARNING] Location connectivity verification failed on attempt {attempt}. Retrying...")
                if not objective_ok:
                    _report(progress_callback, f"[WARNING] Objective completability verification failed on attempt {attempt}. Retrying...")
                world_basic = None
        if world_basic is None:
            raise ValueError("Failed to generate valid world details with proper location connectivity and objective completability after multiple attempts.")
    _log_timing("Paso 3 (mundo base)", step_started)
    _report(progress_callback, f"✅ Mundo base creado con {len(world_basic.locations)} ubicaciones y {len(world_basic.items)} objetos (todas las ubicaciones son accesibles)")

    # Paso 4: Añadir puzzles y obstáculos (with verification)
    _report(progress_callback, "🧩 Paso 4: Añadiendo puzzles y obstáculos...")
    step_started = time.perf_counter()
    if flags['puzzle_stage_only']:
        from .facet_pipeline import run_puzzle_stage
        config = load_config()
        world_with_puzzles = run_puzzle_stage(world_basic, language, model, _parse_range(config['Size_World']['Puzzles']))
    else:
        world_with_puzzles = None
        for attempt in range(1, max_attempts + 1):
            world_with_puzzles = run_step_4_puzzles(world_basic, language, model)
            json_ok = verify_pydantic_model(world_with_puzzles, GeneratedWorld)

            # Validate and fix puzzle rewards
            rewards_ok, world_with_puzzles = verify_puzzle_rewards_and_fix(world_with_puzzles)
            world_with_puzzles = place_unplaced_required_items(world_with_puzzles)

            connectivity_ok = verify_location_connectivity(world_with_puzzles)
            objective_ok = verify_objective_completability(world_with_puzzles)
            if json_ok and rewards_ok and connectivity_ok and objective_ok:
                break
            else:
                if not json_ok:
                    _report(progress_callback, f"[WARNING] Puzzles JSON verification failed on attempt {attempt}. Retrying...")
                if not rewards_ok:
                    _report(progress_callback, f"[WARNING] Puzzle rewards validation failed on attempt {attempt}. Retrying...")
                if not connectivity_ok:
                    _report(progress_callback, f"[WARNING] Puzzles step broke location connectivity on attempt {attempt}. Retrying...")
                if not objective_ok:
                    _report(progress_callback, f"[WARNING] Puzzles step broke objective completability on attempt {attempt}. Retrying...")
                world_with_puzzles = None
        if world_with_puzzles is None:
            raise ValueError("Failed to generate valid puzzles while maintaining location connectivity and objective completability after multiple attempts.")
    _log_timing("Paso 4 (puzzles)", step_started)
    _report(progress_callback, f"✅ Puzzles añadidos: {len(world_with_puzzles.puzzles)} puzzles en total (conectividad preservada)")

    final_msg = "🌱 ¡Generación incremental completada exitosamente!"
    print(final_msg)
    if not validate_world_size(world_with_puzzles):
        print("The generated world does not meet the size requirements.")
    if not verify_location_connectivity(world_with_puzzles):
        print("The generated world does not have all locations connected.")
    if not verify_objective_completability(world_with_puzzles):
        print("The generated world has an objective that cannot be completed with available elements.")
    _log_timing("Generación completa (pasos 1-4)", started)
    if progress_callback:
        progress_callback(final_msg)
    return world_with_puzzles

def skeleton_names_are_unique(skeleton: WorldSkeleton) -> bool:
    """Names identify entities everywhere downstream, so two entities of a kind cannot share one."""
    for entities in (skeleton.key_locations, skeleton.key_items, skeleton.key_characters):
        names = [entity.name for entity in entities]
        if len(names) != len(set(names)):
            print(f"[ERROR] Skeleton has duplicate names: {names}")
            return False
    return True

def validate_world_size(generated_world: GeneratedWorld) -> bool:
    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Parse size parameters
    def parse_range(value):
        if '-' in value:
            return tuple(map(int, value.split('-')))
        return int(value), int(value)

    locations_range = parse_range(config['Size_World']['Locations'])
    objects_range = parse_range(config['Size_World']['Objects'])
    npcs_range = parse_range(config['Size_World']['NPCs'])
    puzzles_range = parse_range(config['Size_World']['Puzzles'])

    # Validate the generated world
    is_valid = (
        locations_range[0] <= len(generated_world.locations) <= locations_range[1] and
        objects_range[0] <= len(generated_world.items) <= objects_range[1] and
        npcs_range[0] <= len(generated_world.characters) <= npcs_range[1] and
        puzzles_range[0] <= len(generated_world.puzzles) <= puzzles_range[1]
    )

    if not is_valid:
        print("[ERROR] Generated world does not adhere to size parameters:")
        print(f"  Locations: {len(generated_world.locations)} (Expected: {locations_range})")
        print(f"  Objects: {len(generated_world.items)} (Expected: {objects_range})")
        print(f"  NPCs: {len(generated_world.characters)} (Expected: {npcs_range})")
        print(f"  Puzzles: {len(generated_world.puzzles)} (Expected: {puzzles_range})")

    return is_valid

def verify_pydantic_model(obj, model_class):
    try:
        if isinstance(obj, model_class):
            return True
        if isinstance(obj, dict):
            model_class.model_validate(obj)
            return True
        if hasattr(obj, 'model_dump'):  # Pydantic v2
            model_class.model_validate(obj.model_dump())
            return True
    except Exception as e:
        print(f"[ERROR] Model verification failed: {e}")
        return False
    return False

def verify_location_connectivity(world: GeneratedWorld) -> bool:
    if not world.locations or len(world.locations) <= 1:
        return True  # Single or no locations are trivially connected
    
    # Build adjacency list including both normal and blocked connections.
    # Edges are added in both directions because create_world_from_llm_response
    # makes every connection bidirectional, so a missing reverse edge is not an error.
    adjacency = {}
    location_names = set()

    for location in world.locations:
        location_names.add(location.name)
        adjacency.setdefault(location.name, set())

        # Add normal connections
        for connected_name in location.connecting_locations:
            adjacency[location.name].add(connected_name)
            adjacency.setdefault(connected_name, set()).add(location.name)

        # Add blocked passages (they're still connections, just blocked)
        for blocked in location.blocked_passages:
            adjacency[location.name].add(blocked.location)
            adjacency.setdefault(blocked.location, set()).add(location.name)
    
    # Perform DFS from the first location to see if we can reach all others
    start_location = next(iter(location_names))
    visited = set()
    stack = [start_location]
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        
        # Add all connected locations to the stack
        for neighbor in adjacency.get(current, set()):
            if neighbor in location_names and neighbor not in visited:
                stack.append(neighbor)
    
    # Check if we visited all locations
    all_connected = len(visited) == len(location_names)
    return all_connected

def verify_objective_completability(world: GeneratedWorld) -> bool:
    if not world.objective:
        print("[ERROR] World has no objective defined")
        return False
    
    objective = world.objective
    obj_type = objective.type.value if hasattr(objective.type, 'value') else str(objective.type)
    
    # Create lookup dictionaries for easier validation
    location_names = {loc.name for loc in world.locations}
    item_names = {item.name for item in world.items}
    character_names = {char.name for char in world.characters}
    
    # Get all items that are accessible (in locations or character inventories)
    accessible_items = set()
    
    # Items in locations
    for location in world.locations:
        accessible_items.update(location.items)
    
    # Items in character inventories
    for character in world.characters:
        accessible_items.update(character.inventory)
    
    # Player inventory
    accessible_items.update(world.player.inventory)
    if obj_type in ["REACH_LOCATION", "reach_location"]:
        # Find the location component
        for component in objective.components:
            component_type = component.component_type.value if hasattr(component.component_type, 'value') else str(component.component_type)
            if component_type in ["LOCATION", "location"]:
                if component.name not in location_names:
                    print(f"[ERROR] REACH_LOCATION objective refers to non-existent location: '{component.name}'")
                    return False
                # Note: Location reachability is already validated by verify_location_connectivity
                return True
        
        print(f"[ERROR] REACH_LOCATION objective has no location component")
        return False
    
    elif obj_type in ["GET_ITEM", "get_item"]:
        # Find all item components and validate each one
        item_components = []
        for component in objective.components:
            component_type = component.component_type.value if hasattr(component.component_type, 'value') else str(component.component_type)
            if component_type in ["ITEM", "item"]:
                item_components.append(component)
        
        if not item_components:
            print(f"[ERROR] GET_ITEM objective has no item component")
            return False
        
        # Validate each item component
        for component in item_components:
            if component.name not in item_names:
                print(f"[ERROR] GET_ITEM objective refers to non-existent item: '{component.name}'")
                return False
            
            # Check if item is accessible (in a location or character inventory)
            if component.name not in accessible_items:
                print(f"[ERROR] GET_ITEM objective item '{component.name}' exists but is not placed anywhere accessible")
                print(f"[ERROR] Item must be in a location or character inventory to be obtainable")
                return False
            
            # Check if item is gettable
            target_item = next((item for item in world.items if item.name == component.name), None)
            if target_item and not target_item.gettable:
                print(f"[ERROR] GET_ITEM objective item '{component.name}' exists but is not gettable")
                return False

        return True
    
    elif obj_type in ["DELIVER_AN_ITEM", "deliver_an_item"]:
        # Need both item and target (location/character) components
        item_component = None
        target_component = None
        
        for component in objective.components:
            component_type = component.component_type.value if hasattr(component.component_type, 'value') else str(component.component_type)
            if component_type in ["ITEM", "item"]:
                item_component = component
            elif component_type in ["LOCATION", "location", "CHARACTER", "character"]:
                target_component = component
        
        if not item_component:
            print(f"[ERROR] DELIVER_AN_ITEM objective has no item component")
            return False
        
        if not target_component:
            print(f"[ERROR] DELIVER_AN_ITEM objective has no target component")
            return False
        
        # Validate item exists and is accessible
        if item_component.name not in item_names:
            print(f"[ERROR] DELIVER_AN_ITEM objective refers to non-existent item: '{item_component.name}'")
            return False
        
        if item_component.name not in accessible_items:
            print(f"[ERROR] DELIVER_AN_ITEM objective item '{item_component.name}' exists but is not placed anywhere accessible")
            return False
        
        # Check if item is gettable
        target_item = next((item for item in world.items if item.name == item_component.name), None)
        if target_item and not target_item.gettable:
            print(f"[ERROR] DELIVER_AN_ITEM objective item '{item_component.name}' exists but is not gettable")
            return False
        
        # Validate target exists
        target_type = target_component.component_type.value if hasattr(target_component.component_type, 'value') else str(target_component.component_type)
        if target_type in ["LOCATION", "location"]:
            if target_component.name not in location_names:
                print(f"[ERROR] DELIVER_AN_ITEM objective refers to non-existent location: '{target_component.name}'")
                return False
        elif target_type in ["CHARACTER", "character"]:
            if target_component.name not in character_names:
                print(f"[ERROR] DELIVER_AN_ITEM objective refers to non-existent character: '{target_component.name}'")
                return False
        
        return True
    
    elif obj_type in ["FIND_CHARACTER", "find_character"]:
        # Find the character component
        for component in objective.components:
            component_type = component.component_type.value if hasattr(component.component_type, 'value') else str(component.component_type)
            if component_type in ["CHARACTER", "character"]:
                if component.name not in character_names:
                    print(f"[ERROR] FIND_CHARACTER objective refers to non-existent character: '{component.name}'")
                    return False
                
                # Check if character is placed in a valid location
                target_character = next((char for char in world.characters if char.name == component.name), None)
                if target_character:
                    if target_character.location not in location_names:
                        print(f"[ERROR] FIND_CHARACTER objective character '{component.name}' is in non-existent location: '{target_character.location}'")
                        return False
                
                return True
        
        print(f"[ERROR] FIND_CHARACTER objective has no character component")
        return False
    
    elif obj_type in ["SOLVE_MYSTERY", "solve_mystery"]:
        # Mirror the clue filter in create_world_from_llm_response: a clue is kept if its item
        # exists and, when item_location is given, the item is actually in that location.
        # Clues are discovered by taking or examining their item, so a kept clue whose item
        # is not in any location or character inventory can never be discovered.
        items_by_location = {location.name: set(location.items) for location in world.locations}
        discoverable_items = set()
        for location in world.locations:
            discoverable_items.update(location.items)
        for character in world.characters:
            discoverable_items.update(character.inventory)

        kept_clues = 0
        for clue in objective.mystery_clues or []:
            if clue.associated_item not in item_names:
                print(f"[WARNING] Mystery clue '{clue.name}' refers to non-existent item '{clue.associated_item}' and will be dropped")
                continue
            if clue.item_location:
                if clue.associated_item not in items_by_location.get(clue.item_location, set()):
                    print(f"[WARNING] Mystery clue '{clue.name}': item '{clue.associated_item}' is not in '{clue.item_location}', the clue will be dropped")
                    continue
            elif clue.associated_item not in discoverable_items:
                print(f"[ERROR] Mystery clue '{clue.name}': item '{clue.associated_item}' is not placed anywhere, so the clue can never be discovered")
                return False
            kept_clues += 1

        if kept_clues == 0:
            print(f"[ERROR] SOLVE_MYSTERY objective has no valid clues")
            return False
        return True
    
    else:
        print(f"[ERROR] Unknown objective type: {obj_type}")
        return False

def place_unplaced_required_items(world: GeneratedWorld) -> GeneratedWorld:
    """Place items the player needs that the LLM left without a position.

    - Items a character gives go to that character's inventory.
    - Items required to unblock a passage or requested by a character go to the
      player's starting location, which is always reachable.
    Objective items are left untouched: placing them at the start would make the
    objective trivial, so verify_objective_completability still rejects them.
    """
    item_names = {item.name for item in world.items}

    placed_items = set(world.player.inventory)
    for location in world.locations:
        placed_items.update(location.items)
    for character in world.characters:
        placed_items.update(character.inventory)

    objective_items = {item.name for item in world.items if item.is_objective_target}
    if world.objective:
        for component in world.objective.components:
            component_type = component.component_type.value if hasattr(component.component_type, 'value') else str(component.component_type)
            if component_type.lower() == "item":
                objective_items.add(component.name)

    def required_item_name(requirement):
        req_type = requirement.requirement_type.value if hasattr(requirement.requirement_type, 'value') else str(requirement.requirement_type)
        if req_type.lower() == 'item':
            return getattr(requirement, 'item_name', None)
        return None

    # Items given by characters go to the giver's inventory
    for character in world.characters:
        given_item = character.interaction.gives_item if character.interaction else None
        if given_item in item_names and given_item not in placed_items:
            character.inventory.append(given_item)
            placed_items.add(given_item)
            print(f"[INFO] Item '{given_item}' given by '{character.name}' was not placed anywhere. Added it to their inventory")

    # Keys for blocked passages and items requested by characters
    required_items = []
    for location in world.locations:
        for blocked in location.blocked_passages:
            required_items.append(required_item_name(blocked.required_to_unblock))
    for character in world.characters:
        if character.interaction:
            for requirement in character.interaction.requires:
                required_items.append(required_item_name(requirement))

    # Same starting location the world builder uses
    start_location = next((loc for loc in world.locations if loc.name == world.player.location), None)
    if start_location is None and world.locations:
        start_location = world.locations[0]

    for item_name in required_items:
        if (item_name in item_names and item_name not in placed_items
                and item_name not in objective_items and start_location):
            start_location.items.append(item_name)
            placed_items.add(item_name)
            print(f"[WARNING] Required item '{item_name}' was not placed anywhere. Added it to starting location '{start_location.name}'")

    return world

def verify_puzzle_rewards_and_fix(world: GeneratedWorld) -> tuple[bool, GeneratedWorld]:
    from .structured_data_models import GeneratedItem, ItemReward, ItemActionType

    # Track changes made
    items_created = []
    
    # Get existing item names for quick lookup
    existing_item_names = {item.name for item in world.items}
    existing_item_names_lower = {name.lower() for name in existing_item_names}
    
    for puzzle in world.puzzles:
        
        for reward in puzzle.rewards:
            if isinstance(reward, ItemReward):
                item_name = reward.item_name
                
                if item_name.lower() not in existing_item_names_lower:
                    print(f"[WARNING] ItemReward item '{item_name}' does not exist. Creating it...")
                    
                    # Create the missing item
                    new_item = GeneratedItem(
                        name=item_name,
                        action_type=ItemActionType.LORE,  # required field; not used by the game engine
                        descriptions=[f"A {item_name.lower()} obtained as a reward for solving puzzles."],
                        gettable=True,
                        is_objective_target=False,
                        relevance_to_objective=f"Reward item from puzzle '{puzzle.name}'",
                        required_for=[]
                    )
                    
                    world.items.append(new_item)
                    existing_item_names.add(item_name)
                    items_created.append(item_name)
                    
                    # Determine where to place the item
                    # For observation puzzles, the item should be in the location, not on the character
                    if puzzle.puzzle_type == "observation" and puzzle.location:
                        location_found = False
                        for location in world.locations:
                            if location.name == puzzle.location:
                                location.items.append(item_name)
                                print(f"[INFO] Added observation puzzle reward item '{item_name}' to location '{location.name}'")
                                location_found = True
                                break
                        if not location_found:
                             print(f"[WARNING] Puzzle location '{puzzle.location}' not found, adding item to first location")
                             if world.locations:
                                 world.locations[0].items.append(item_name)

                    elif puzzle.proposed_by_character:
                        # For other puzzle types, add to character's inventory
                        char_found = False
                        for character in world.characters:
                            if character.name == puzzle.proposed_by_character:
                                character.inventory.append(item_name)
                                print(f"[INFO] Added item '{item_name}' to character '{character.name}' inventory")
                                char_found = True
                                break
                        
                        if not char_found:
                            print(f"[WARNING] Character '{puzzle.proposed_by_character}' not found, adding item to first location")
                            if world.locations:
                                world.locations[0].items.append(item_name)
                    
                    elif puzzle.location:
                        # Add to puzzle location
                        location_found = False
                        for location in world.locations:
                            if location.name == puzzle.location:
                                location.items.append(item_name)
                                print(f"[INFO] Added item '{item_name}' to location '{location.name}'")
                                location_found = True
                                break
                        
                        if not location_found:
                            print(f"[WARNING] Puzzle location '{puzzle.location}' not found, adding item to first location")
                            if world.locations:
                                world.locations[0].items.append(item_name)
                    
                    else:
                        # Add to first location as fallback
                        if world.locations:
                            world.locations[0].items.append(item_name)
                            print(f"[INFO] Added item '{item_name}' to location '{world.locations[0].name}' (fallback)")
        
    return True, world
