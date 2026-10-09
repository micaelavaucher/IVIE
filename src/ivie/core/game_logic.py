"""Game logic and world management for IVIE.

This module contains the core game loop, world state management,
and game-related utility functions with intelligent memory integration.
"""

import re
import jsonpickle
import time
import streamlit as st

from ..config import load_config
from ..llm.prompts import prompt_narrate_current_scene, prompt_world_update_structured
from ..llm.turn_prompts import prompt_interpret_action, prompt_narrate_outcome
from .narration_check import narration_problems, snapshot
from .turn_engine import apply_turn
from .world_builder import generate_world_overview, generate_objective_validation_report
from .world_utils import create_world_state_summary
from ..llm.structured_data_models import WorldUpdate
from ..llm.memory_system import create_memory_system
from ..database.mongodb_handler import db_handler

def _narrate_opening_from_facts(world, language, narrative_model):
    """The opening scene written like any turn: from the engine's facts (where the player starts and
    what they carry), so it cannot put in the player's pocket an item lying in the room."""
    from .turn_engine import TurnOutcome
    location, inventory = world.player.location, [item.name for item in world.player.inventory]
    facts = TurnOutcome()
    facts.add('start', True, f"El jugador comienza la aventura en {location.name}",
              f"The player begins the adventure in {location.name}")
    facts.add('inventory', True,
              f"El jugador lleva: {', '.join(inventory)}" if inventory else "El jugador no lleva ningún objeto",
              f"The player carries: {', '.join(inventory)}" if inventory else "The player carries no items")
    system_msg, user_msg = prompt_narrate_outcome(
        world.render_world(language=language, reveal_answers=False, for_narrator=True),
        "(comienzo de la aventura)" if language == 'es' else "(start of the adventure)",
        "", facts.render(language), [], arrived=True, first_visit=True,
        objective=world.objective_statement(language) if world.objective else "",
        objective_completed=False, language=language)
    return narrative_model.prompt_model(system_msg=system_msg, user_msg=user_msg)

def generate_starting_narration(world, language, narrative_model):
    started = time.perf_counter()
    if narrate_from_outcome_enabled():
        starting_narration = _narrate_opening_from_facts(world, language, narrative_model)
    else:
        system_msg_current_scene, user_msg_current_scene = prompt_narrate_current_scene(
            world.render_world(language=language),
            previous_narrations=world.player.visited_locations[world.player.location.name],
            language=language,
            starting_scene=True,
        )
        starting_narration = narrative_model.prompt_model(system_msg=system_msg_current_scene, user_msg=user_msg_current_scene)
    world.player.visited_locations[world.player.location.name] += [starting_narration]

    objective = world.objective_statement(language) if getattr(world, 'objective', None) else ""
    if objective:
        starting_narration += f"\n\n🎯 {objective}"
    else:
        print("ℹ️ No se encontró un objetivo principal en el mundo generado/cargado.")

    world_state_formatted = world.format_world_state_for_chat(language=language)
    starting_narration += f"\n\n---\n{world_state_formatted}"
    print(f"[TIMING] Narración inicial: {time.perf_counter() - started:.1f}s")
    return starting_narration

def create_game_log_entry(language, narrative_model_name, reasoning_model_name, world_id=None, inspiration=""):
    game_log_dictionary = {}
    game_log_dictionary["nickname"] = st.session_state.nickname
    game_log_dictionary["language"] = language
    game_log_dictionary["world_id"] = world_id or f"generated_{int(time.time())}"
    game_log_dictionary["narrative_model_name"] = narrative_model_name
    game_log_dictionary["reasoning_model_name"] = reasoning_model_name
    game_log_dictionary["inspiration"] = inspiration
    return game_log_dictionary

def generate_world_overview(world, language):
    overview = ""
    header = "🌎 Resumen del Mundo 🌍\n\n" if language == 'es' else "🌎 World Overview 🌍\n\n"
    overview += header

    locations = world.locations if (hasattr(world, 'locations') and world.locations) else {world.player.location.name: world.player.location}
    if not (hasattr(world, 'locations') and world.locations) and hasattr(world, 'characters') and world.characters:
        for char in world.characters.values():
            if hasattr(char, 'location') and char.location:
                locations[char.location.name] = char.location

    for location_name, location in locations.items():
        overview += f"{location_name}:\n"

        connections = [loc.name for loc in location.connecting_locations] if (hasattr(location, 'connecting_locations') and location.connecting_locations) else []
        if connections:
            overview += (" - Conecta con:\n" if language == 'es' else " - Connects to:\n")
            for conn in connections:
                overview += f"   - {conn}\n"
        else:
            overview += (" - Conecta con: Ninguna\n" if language == 'es' else " - Connects to: None\n")

        items = [item.name for item in location.items] if (hasattr(location, 'items') and location.items) else []
        if items:
            overview += (" - Objetos:\n" if language == 'es' else " - Items:\n")
            for item in items:
                overview += f"   - {item}\n"
        else:
            overview += (" - Objetos: Ninguno\n" if language == 'es' else " - Items: None\n")

        npcs_with_inventory = []
        if hasattr(world, 'characters') and world.characters:
            for char in world.characters.values():
                if (hasattr(char, 'location') and char.location is location and char is not world.player):
                    char_inventory = [item.name for item in char.inventory] if (hasattr(char, 'inventory') and char.inventory) else []
                    npcs_with_inventory.append((char.name, char_inventory))

        if npcs_with_inventory:
            overview += (" - PNJs:\n" if language == 'es' else " - NPCs:\n")
            for npc_name, npc_inventory in npcs_with_inventory:
                overview += f"   - {npc_name}"
                if npc_inventory:
                    overview += (f" (Inventario: {', '.join(npc_inventory)})" if language == 'es' else f" (Inventory: {', '.join(npc_inventory)})")
                else:
                    overview += (" (Inventario: Vacío)" if language == 'es' else " (Inventory: Empty)")
                overview += "\n"
        else:
            overview += (" - PNJs: Ninguno\n" if language == 'es' else " - NPCs: None\n")

        overview += "\n"

    return overview

def get_objective_info(world, language):
    if not hasattr(world, 'objective') or not world.objective:
        return "🎯 No se ha definido un objetivo específico para este mundo." if language == 'es' else "🎯 No specific objective has been defined for this world."

    raw_objective = ""

    # Handle different objective formats
    if hasattr(world.objective, 'description'):
        # New structured objective format - this should be used for newer worlds
        raw_objective = world.objective.description
    elif isinstance(world.objective, tuple) and len(world.objective) >= 2:
        # Legacy tuple format
        obj_component = world.objective[1]
        # MysteryObjective and ItemCollection also have a description; they are handled below, with their progress
        if hasattr(obj_component, 'description') and obj_component.__class__.__name__ not in ('MysteryObjective', 'ItemCollection'):
            raw_objective = obj_component.description
        elif hasattr(obj_component, '__class__') and obj_component.__class__.__name__ == 'MysteryObjective':
            # Special handling for mystery objectives
            mystery_obj = obj_component
            # The sentence states the real number of clues; the generated description may not
            raw_objective = f"🎯 {world.objective_statement(language)}" + chr(10) * 2 + mystery_obj.description
            discovered, total = mystery_obj.get_completion_progress()
            if language == 'es':
                raw_objective += f"\n\n🔍 **Progreso del Misterio:** {discovered}/{total} pistas descubiertas"
                discovered_clues = mystery_obj.get_discovered_clues()
                if discovered_clues:
                    raw_objective += "\n\n📚 **Pistas Descubiertas:**"
                    for clue in discovered_clues:
                        raw_objective += f"\n• **{clue.name}:** {clue.description}"
                        raw_objective += f"\n  *Relevancia:* {clue.relevance_to_mystery}"
                remaining = total - discovered
                if remaining > 0:
                    raw_objective += f"\n\n🔍 Quedan {remaining} pistas por descubrir. Interactúa con objetos para encontrarlas."
                elif getattr(mystery_obj, 'culprit', None) and not mystery_obj.is_completed():
                    raw_objective += "\n\n⚖️ Ya tienes todas las pistas: di quién es el responsable."
            else:
                raw_objective += f"\n\n🔍 **Mystery Progress:** {discovered}/{total} clues discovered"
                discovered_clues = mystery_obj.get_discovered_clues()
                if discovered_clues:
                    raw_objective += "\n\n📚 **Discovered Clues:**"
                    for clue in discovered_clues:
                        raw_objective += f"\n• **{clue.name}:** {clue.description}"
                        raw_objective += f"\n  *Relevance:* {clue.relevance_to_mystery}"
                remaining = total - discovered
                if remaining > 0:
                    raw_objective += f"\n\n🔍 {remaining} clues remain to be discovered. Interact with objects to find them."
                elif getattr(mystery_obj, 'culprit', None) and not mystery_obj.is_completed():
                    raw_objective += "\n\n⚖️ You have every clue: say who is responsible."
        else:
            # Enhanced handling for different objective types based on tuple structure
            player_or_item = world.objective[0]
            target = world.objective[1]
            
            if "ItemCollection" in (player_or_item.__class__.__name__, target.__class__.__name__):
                # Several items needed together: the sentence plus how many are already in place
                collection = target if target.__class__.__name__ == "ItemCollection" else player_or_item
                container = (world.player.inventory if collection is target
                             else target.items if target.__class__.__name__ == "Location" else target.inventory)
                done, total = collection.progress(container)
                raw_objective = f"🎯 {world.objective_statement(language)}"
                raw_objective += (f"\n\n📦 Progreso: {done}/{total}" if language == 'es' else f"\n\n📦 Progress: {done}/{total}")
            # Check if this is a delivery objective (Item -> Location/Character)
            elif player_or_item.__class__.__name__ == "Item":
                # DELIVER_AN_ITEM objective: (item, target_location_or_character)
                item = player_or_item
                target_class = target.__class__.__name__
                
                if target_class == "Location":
                    if language == 'es':
                        raw_objective = f"📦 Entregar el objeto **{item.name}** a la ubicación: **{target.name}**"
                    else:
                        raw_objective = f"📦 Deliver the item **{item.name}** to location: **{target.name}**"
                        
                            
                elif target_class == "Character":
                    if language == 'es':
                        raw_objective = f"📦 Entregar el objeto **{item.name}** al personaje: **{target.name}**"
                       
                    else:
                        raw_objective = f"📦 Deliver the item **{item.name}** to character: **{target.name}**"
                       
                else:
                    # Fallback for unknown delivery target type
                    if language == 'es':
                        raw_objective = f"📦 Entregar el objeto **{item.name}** a: **{target.name}**"
                    else:
                        raw_objective = f"📦 Deliver the item **{item.name}** to: **{target.name}**"
            else:
                # Standard player-based objectives (GET_ITEM, REACH_LOCATION, FIND_CHARACTER)
                player = player_or_item
                target_class = target.__class__.__name__
                
                if target_class == "Item":
                    # GET_ITEM objective
                    if language == 'es':
                        raw_objective = f"🎒 Encontrar y obtener el objeto: **{target.name}**"

                    else:
                        raw_objective = f"🎒 Find and obtain the item: **{target.name}**"
                        
                elif target_class == "Location":
                    # REACH_LOCATION objective
                    if language == 'es':
                        raw_objective = f"📍 Llegar a la ubicación: **{target.name}**"
                    else:
                        raw_objective = f"📍 Reach the location: **{target.name}**"
                       
                elif target_class == "Character":
                    # FIND_CHARACTER objective
                    if language == 'es':
                        raw_objective = f"👤 Encontrar al personaje: **{target.name}**"
                    else:
                        raw_objective = f"👤 Find the character: **{target.name}**"
                        
                else:
                    # Fallback for unknown target types
                    if language == 'es':
                        raw_objective = f"🎯 Completar la tarea relacionada con: **{target.name}**"
                    else:
                        raw_objective = f"🎯 Complete the task related to: **{target.name}**"
                    
    elif isinstance(world.objective, str):
        # Simple string objective
        raw_objective = world.objective
    else:
        # Fallback for unknown formats
        if language == 'es':
            raw_objective = "🎯 Objetivo no especificado claramente."
        else:
            raw_objective = "🎯 Objective not clearly specified."
    
    return raw_objective

def check_character_puzzle_mention(world, message, language):
    if not message:
        return None
    message_lower = message.lower()

    for character in world.characters.values():
        if character.location == world.player.location:
            character_name_lower = character.name.lower()
            if character_name_lower in message_lower:
                if not (hasattr(character, 'interaction') and character.interaction and hasattr(character.interaction, 'proposes_puzzle') and character.interaction.proposes_puzzle):
                    return None

                puzzle_name = character.interaction.proposes_puzzle
                if puzzle_name in world.puzzle_states and world.puzzle_states[puzzle_name] == 'solved':
                    return None

                hint_keywords = ['hint', 'pista', 'help', 'ayuda', 'clue', 'stuck', 'atascado']
                if any(k in message_lower for k in hint_keywords):
                    return None

                solve_keywords = ['answer', 'respuesta', 'solve', 'resolver', 'solution', 'solución']
                question_keywords = ['about', 'sobre', 'tell me', 'dime', 'what', 'qué', 'who', 'quién', 'why', 'por qué']
                is_question = any(k in message_lower for k in question_keywords)
                is_solving = any(k in message_lower for k in solve_keywords)
                if is_question and not is_solving:
                    return None

                if puzzle_name in world.puzzles:
                    puzzle = world.puzzles[puzzle_name]
                    if language == 'es':
                        response = f"🎭 **{character.name}** se acerca a ti"
                        if hasattr(character.interaction, 'interaction_text') and character.interaction.interaction_text:
                            response += f" y dice: \"{character.interaction.interaction_text}\""
                        response += f"\n\n🧩 **Puzzle: {puzzle.name}**\n"
                        response += f"📝 **Problema:** {puzzle.problem}\n"
                        if hasattr(puzzle, 'puzzle_hints') and puzzle.puzzle_hints:
                            response += f"💡 **Pistas disponibles:** {len(puzzle.puzzle_hints)} pistas"
                        response += f"\n\n⚠️ *Debes resolver este puzzle antes de que {character.name} pueda ayudarte.*"
                    else:
                        response = f"🎭 **{character.name}** approaches you"
                        if hasattr(character.interaction, 'interaction_text') and character.interaction.interaction_text:
                            response += f" and says: \"{character.interaction.interaction_text}\""
                        response += f"\n\n🧩 **Puzzle: {puzzle.name}**\n"
                        response += f"📝 **Problem:** {puzzle.problem}\n"
                        if hasattr(puzzle, 'puzzle_hints') and puzzle.puzzle_hints:
                            response += f"💡 **Hints available:** {len(puzzle.puzzle_hints)} hints"
                        response += f"\n\n⚠️ *You must solve this puzzle before {character.name} can help you.*"

                    print(f"🧩 Character puzzle proposition triggered: {character.name} -> {puzzle.name}")
                    return response
    return None

# Commands are recognized by the whole message, not by a word inside it: "examino la Pista del
# Zafiro" is an action on an item called "Pista del Zafiro", not a request for help
HINT_COMMANDS = {
    "pista", "ayuda", "una pista", "dame una pista", "pido una pista", "quiero una pista", "necesito una pista",
    "necesito ayuda", "ayudame", "hint", "help", "clue", "a hint", "give me a hint", "i need a hint",
    "i need help", "help me",
}


def _command(message):
    from ..llm.name_matching import normalize_name
    return " ".join(normalize_name(message or "").split())


def handle_debug_command(message, world, language):
    message_lower = message.lower()
    if _command(message) in HINT_COMMANDS:
        hint = world.get_next_hint(language)
        return f"💡 **Pista:** {hint}" if language == 'es' else f"💡 **Hint:** {hint}"

    if message_lower in ["inspect", "inspeccionar", "inspect world", "inspeccionar mundo"]:
        overview = generate_world_overview(world, language)
        validation = generate_objective_validation_report(world, language)
        debug_info = overview + "\n\n" + validation
        return debug_info.replace("<", r"\<").replace(">", r"\>")

    if message_lower in ["see world", "ver mundo", "world overview", "resumen mundo", "overview"]:
        world_overview = generate_world_overview(world, language)
        return world_overview.replace("<", r"\<").replace(">", r"\>")

    if message_lower in ["validate objective", "validar objetivo", "check objective", "objetivo válido", "objetivo valido", "valid objective"]:
        validation_report = generate_objective_validation_report(world, language)
        return validation_report.replace("<", r"\<").replace(">", r"\>")

    if message_lower in ["objective", "objetivo", "what is my objective", "what is my objective?", "cuál es mi objetivo", "cuál es mi objetivo?", "cual es mi objetivo", "cual es mi objetivo?", "my objective", "mi objetivo", "goal", "meta"]:
        objective_info = get_objective_info(world, language)
        return objective_info.replace("<", r"\<").replace(">", r"\>")

    return None

def process_player_input_structured(world, message, language, reasoning_model, number_of_turns, game_log_dictionary, memory_system=None):
    game_log_dictionary[number_of_turns]["date"] = time.ctime(time.time())
    game_log_dictionary[number_of_turns]["previous_symbolic_world_state"] = jsonpickle.encode(world, unpicklable=False)
    game_log_dictionary[number_of_turns]["previous_rendered_world_state"] = world.render_world(language=language)
    game_log_dictionary[number_of_turns]["user_input"] = message

    relevant_memories_text = ""
    if memory_system:
        try:
            step_started = time.perf_counter()
            relevant_memories = memory_system.retrieve_relevant_memories(message, top_k=3)
            print(f"[TIMING] turno: recuperar memorias {time.perf_counter() - step_started:.1f}s")
            relevant_memories_text = memory_system.format_memories_for_prompt(relevant_memories, language)
        except Exception as e:
            print(f"⚠️ Error retrieving memories: {e}")

    system_msg_update, user_msg_update, expected_model = prompt_world_update_structured(
        world.render_world(language=language), message, language=language, relevant_memories=relevant_memories_text
    )

    try:
        full_prompt = system_msg_update + "\n\n" + user_msg_update
        response_schema = expected_model.model_json_schema()
        step_started = time.perf_counter()
        response_json = reasoning_model.prompt_model_structured(prompt=full_prompt, response_schema=response_schema)
        print(f"[TIMING] turno: razonador {time.perf_counter() - step_started:.1f}s")

        if isinstance(response_json, dict):
            world_update = WorldUpdate(**response_json)
        elif isinstance(response_json, str):
            import json
            response_data = json.loads(response_json)
            world_update = WorldUpdate(**response_data)
        else:
            world_update = response_json

        print("🛠️ Predicted outcomes of the player input 🛠️")
        print(f"> Player input: {message}")
        print(f"{world_update.narration}\n")

        game_log_dictionary[number_of_turns]["predicted_outcomes"] = world_update.narration
        game_log_dictionary[number_of_turns]["structured_update"] = world_update.model_dump()

        world.update_from_structured(world_update, language=language, player_input=message)

        if memory_system:
            try:
                world_state_summary = create_world_state_summary(world, message, language)
                step_started = time.perf_counter()
                memory_system.ingest_memory(turn_number=number_of_turns, player_action=message, result_narration=world_update.narration, world_state_summary=world_state_summary)
                print(f"[TIMING] turno: guardar memoria {time.perf_counter() - step_started:.1f}s")
            except Exception as e:
                print(f"⚠️ Error ingesting memory: {e}")

    except Exception as e:
        print(f"⚠️ Structured processing failed: {e}")
        print("🔄 Falling back to legacy text processing...")
        response_update = reasoning_model.prompt_model(system_msg=system_msg_update, user_msg=user_msg_update)

        print("🛠️ Predicted outcomes of the player input 🛠️")
        print(f"> Player input: {message}")
        try:
            predicted_outcomes = re.sub(r'#([^#]*?)#', '', response_update)
            print(f"{predicted_outcomes}\n")
            game_log_dictionary[number_of_turns]["predicted_outcomes"] = predicted_outcomes
        except Exception as e:
            print(f"Error: {e}")

        world.update(response_update)
        game_log_dictionary[number_of_turns]["fallback_response"] = response_update

        if memory_system:
            try:
                narration_matches = re.findall(r'#([^#]*?)#', str(response_update))
                fallback_narration = narration_matches[0] if narration_matches else "Something happened in the world."
                world_state_summary = create_world_state_summary(world, message, language)
                memory_system.ingest_memory(turn_number=number_of_turns, player_action=message, result_narration=fallback_narration, world_state_summary=world_state_summary)
            except Exception as e:
                print(f"⚠️ Error ingesting fallback memory: {e}")

    game_log_dictionary[number_of_turns]["updated_symbolic_world_state"] = jsonpickle.encode(world, unpicklable=True)
    game_log_dictionary[number_of_turns]["updated_rendered_world_state"] = world.render_world(language=language)

    return world_update.narration if 'world_update' in locals() else (response_update if 'response_update' in locals() else "")

def generate_narration(world, last_player_position, response_update, language, narrative_model):
    answer = ""
    if last_player_position is not world.player.location:
        is_first_visit = not world.player.location.visited
        system_msg_new_scene, user_msg_new_scene = prompt_narrate_current_scene(
            world.render_world(language=language),
            previous_narrations=world.player.visited_locations[world.player.location.name],
            language=language,
            first_visit=is_first_visit,
        )
        new_scene_narration = narrative_model.prompt_model(system_msg=system_msg_new_scene, user_msg=user_msg_new_scene)
        world.player.visited_locations[world.player.location.name] += [new_scene_narration]
        answer += f"\n{new_scene_narration}\n\n"
        if is_first_visit:
            world.player.location.visited = True
        last_player_position = world.player.location
    else:
        try:
            if not response_update or not response_update.strip():
                answer += "Algo sucedió en el mundo.\n" if language == 'es' else "Something happened in the world.\n"
            elif isinstance(response_update, str) and not response_update.startswith(("- Moved object:", "- Blocked passages", "- Your location")):
                answer += f"{response_update}\n"
            else:
                narration_matches = re.findall(r'#([^#]*?)#', str(response_update))
                answer += f"{narration_matches[0]}\n" if narration_matches else ("Algo sucedió en el mundo.\n" if language == 'es' else "Something happened in the world.\n")
        except Exception as e:
            print(f"Error extracting narration: {e}")
            answer += "Algo sucedió en el mundo.\n" if language == 'es' else "Something happened in the world.\n"

    return answer, last_player_position

def objective_completed(world) -> bool:
    return bool(getattr(world, 'objective', None)) and world.check_objective()


def check_objective_completion(world, answer, language, include_completion_text=True, completed_before=False):
    """Append the completion banner on the turn the objective is completed (not on every later
    turn). With include_completion_text=False the pre-written ending is left out: the narrator
    already wrote one from what happened, while the pre-written text may describe steps the game
    never asked for ("with the capacitor reprogrammed..." or a culprit who is not in the world)."""
    if not completed_before and world.check_objective():
        answer += "\n\n🎯¡Completaste el objetivo!" if language == 'es' else "\n\n🎯You have completed your quest!"
        if not include_completion_text:
            return answer

        if hasattr(world, 'objective_data') and world.objective_data:
            obj_type = world.objective_data.type.value if hasattr(world.objective_data.type, 'value') else str(world.objective_data.type)
            if obj_type in ["SOLVE_MYSTERY", "solve_mystery"] and hasattr(world.objective_data, 'mystery_solution') and world.objective_data.mystery_solution:
                answer += f"\n\n🎭 **Solución del Misterio:**\n{world.objective_data.mystery_solution}" if language == 'es' else f"\n\n🎭 **Mystery Solution:**\n{world.objective_data.mystery_solution}"
            elif hasattr(world.objective_data, 'completion_narration') and world.objective_data.completion_narration:
                answer += f"\n\n📖 **Final:**\n{world.objective_data.completion_narration}" if language == 'es' else f"\n\n📖 **Conclusion:**\n{world.objective_data.completion_narration}"

        elif (hasattr(world, 'objective') and world.objective and isinstance(world.objective, tuple) and len(world.objective) >= 2):
            obj_component = world.objective[1]
            if (hasattr(obj_component, '__class__') and obj_component.__class__.__name__ == 'MysteryObjective' and hasattr(obj_component, 'mystery_solution')):
                answer += f"\n\n🎭 **Solución del Misterio:**\n{obj_component.mystery_solution}" if language == 'es' else f"\n\n🎭 **Mystery Solution:**\n{obj_component.mystery_solution}"
            elif hasattr(obj_component, 'completion_narration') and obj_component.completion_narration:
                answer += f"\n\n📖 **Final:**\n{obj_component.completion_narration}" if language == 'es' else f"\n\n📖 **Conclusion:**\n{obj_component.completion_narration}"

    return answer

def save_game_log(game_log_dictionary, number_of_turns, answer):
    game_log_dictionary[number_of_turns]["narration"] = answer
    world_id = game_log_dictionary.get("world_id")
    current_turn_data = game_log_dictionary.get(number_of_turns)
    if db_handler and world_id and current_turn_data:
        db_handler.add_turn_to_trace(world_id, number_of_turns, current_turn_data)

def check_narration_enabled(config=None) -> bool:
    """[Game] CheckNarration: check each narration against the world and rewrite it once if it contradicts it."""
    config = config or load_config()
    return config.getboolean('Game', 'CheckNarration', fallback=True)


def narrate_from_outcome_enabled(config=None) -> bool:
    """[Game] NarrateFromOutcome: the engine decides each turn and the narration is written from what happened."""
    config = config or load_config()
    return config.getboolean('Game', 'NarrateFromOutcome', fallback=False)

def play_turn_from_outcome(world, message, language, reasoning_model, narrative_model, number_of_turns, game_log_dictionary, memory_system=None):
    """One turn in three steps: the reasoning model interprets the action (TurnIntent), the engine
    applies what the rules allow (turn_engine.apply_turn), and the narrative model narrates the
    engine's facts. Returns the narration."""
    completed_before = objective_completed(world)
    log = game_log_dictionary[number_of_turns]
    log["date"] = time.ctime(time.time())
    log["previous_symbolic_world_state"] = jsonpickle.encode(world, unpicklable=False)
    log["user_input"] = message

    relevant_memories_text = ""
    if memory_system:
        try:
            step_started = time.perf_counter()
            relevant_memories = memory_system.retrieve_relevant_memories(message, top_k=3)
            relevant_memories_text = memory_system.format_memories_for_prompt(relevant_memories, language)
            print(f"[TIMING] turno: recuperar memorias {time.perf_counter() - step_started:.1f}s")
        except Exception as e:
            print(f"⚠️ Error retrieving memories: {e}")

    # 1. Interpret: what does the player attempt?
    system_msg, user_msg, intent_model = prompt_interpret_action(
        world.render_world(language=language, reveal_answers=False), message, language, relevant_memories_text,
        characters=[c.name for c in world.characters.values() if c is not world.player])
    step_started = time.perf_counter()
    try:
        response = reasoning_model.prompt_model_structured(prompt=system_msg + "\n\n" + user_msg, response_schema=intent_model)
        intent = intent_model.model_validate(response) if isinstance(response, dict) and response else intent_model(action_summary=message)
    except Exception as e:
        print(f"⚠️ Could not interpret the action, treating it as having no effect: {e}")
        intent = intent_model(action_summary=message)
    print(f"[TIMING] turno: razonador {time.perf_counter() - step_started:.1f}s")

    # 2. Apply: the engine decides what actually happens
    before = snapshot(world)
    outcome = apply_turn(world, intent, message)
    just_completed = not completed_before and objective_completed(world)
    print(f"🛠️ Engine facts for '{message}':\n{outcome.render(language)}")

    # 3. Narrate the facts, and rewrite once a narration that contradicts the world
    location = world.player.location

    def narrate(corrections=None):
        system_msg, user_msg = prompt_narrate_outcome(
            world.render_world(language=language, reveal_answers=False, for_narrator=True), message, intent.action_summary,
            outcome.render(language), world.player.visited_locations.get(location.name, []),
            arrived=outcome.arrived_at is not None, first_visit=outcome.first_visit,
            objective=world.objective_statement(language) if world.objective else "",
            objective_completed=just_completed, language=language, completed_before=completed_before,
            corrections=corrections)
        step_started = time.perf_counter()
        text = narrative_model.prompt_model(system_msg=system_msg, user_msg=user_msg)
        print(f"[TIMING] turno: narrador {time.perf_counter() - step_started:.1f}s")
        return text

    narration = narrate()
    if check_narration_enabled():
        step_started = time.perf_counter()
        problems = narration_problems(world, narration, before, outcome, objective_completed(world), language, reasoning_model)
        print(f"[TIMING] turno: control de la narración {time.perf_counter() - step_started:.1f}s")
        if problems:
            print(f"🔁 Narration contradicted the world, rewriting it: {problems}")
            log["rejected_narration"] = narration
            log["narration_problems"] = problems
            narration = narrate(corrections=problems)
    world.player.visited_locations.setdefault(location.name, []).append(narration)
    if outcome.arrived_at is not None:
        location.visited = True

    log["structured_intent"] = intent.model_dump()
    log["engine_facts"] = [event.text(language) for event in outcome.events]
    log["predicted_outcomes"] = narration
    log["updated_symbolic_world_state"] = jsonpickle.encode(world, unpicklable=True)
    log["updated_rendered_world_state"] = world.render_world(language=language)

    if memory_system:
        try:
            step_started = time.perf_counter()
            world_state_summary = create_world_state_summary(world, message, language)
            # The engine's facts are what happened; the narration only adds dialogue and colour
            facts = "; ".join(event.text(language) for event in outcome.events) or (
                "ningún cambio" if language == 'es' else "no change")
            remembered = (f"Hechos: {facts}. Narración: {narration}" if language == 'es'
                          else f"Facts: {facts}. Narration: {narration}")
            memory_system.ingest_memory(turn_number=number_of_turns, player_action=message, result_narration=remembered, world_state_summary=world_state_summary)
            print(f"[TIMING] turno: guardar memoria {time.perf_counter() - step_started:.1f}s")
        except Exception as e:
            print(f"⚠️ Error ingesting memory: {e}")

    return narration

def create_game_loop(world, reasoning_model, narrative_model, language, visited_locations, api_key=None, enable_rag=True, inspiration="",
                     world_id=None, start_turn=0, narrate_from_outcome=None):
    """world_id and start_turn let a session resumed in a new process keep its RAG memories and turn count.
    narrate_from_outcome selects the turn pipeline; None reads it from config.ini."""
    if narrate_from_outcome is None:
        narrate_from_outcome = narrate_from_outcome_enabled()
    print(f"⚙️ Turn pipeline: {'engine decides, narration from outcome' if narrate_from_outcome else 'single model call'}")
    last_player_position = world.player.location
    number_of_turns = start_turn
    
    world.update_hints()
    session_world_id = world_id or f"generated_{int(time.time())}"
    
    game_log_dictionary = create_game_log_entry(
        language,
        narrative_model.model_name if hasattr(narrative_model, 'model_name') else 'unknown',
        reasoning_model.model_name if hasattr(reasoning_model, 'model_name') else 'unknown',
        world_id=session_world_id,
        inspiration=inspiration,
    )
    
    # turn 0 with world state
    game_log_dictionary[0] = {
        "date": time.ctime(time.time()),
        "initial_symbolic_world_state": jsonpickle.encode(world, unpicklable=True),
        "initial_rendered_world_state": world.render_world(language=language),
    }
    
    # MongoDB trace document
    if db_handler:
        mongo_document = {
            "world_id": session_world_id,
            "nickname": game_log_dictionary["nickname"],
            "language": language,
            "narrative_model_name": narrative_model.model_name if hasattr(narrative_model, 'model_name') else 'unknown',
            "reasoning_model_name": reasoning_model.model_name if hasattr(reasoning_model, 'model_name') else 'unknown',
            "inspiration": inspiration,
            "created_at": time.time(),
            "turns": {
                "0": game_log_dictionary[0]
            }
        }
        try:
            db_handler.initialize_trace(mongo_document)
            print(f"💾 Game trace initialized in MongoDB with world_id: {session_world_id}")
        except Exception as e:
            print(f"⚠️ Failed to initialize MongoDB trace: {e}")

    # Initialize memory system if RAG is enabled
    memory_system = None
    if enable_rag:
        try:
            memory_system = create_memory_system(session_world_id, api_key)
            print(f"🧠 Intelligent memory system initialized for world {session_world_id}")
            
            # Load existing memories from previous logs if available. A resumed session (world_id given)
            # already has its memories in the persistent vector store; loading them again from the
            # trace would duplicate them
            if session_world_id and world_id is None:
                memory_system.load_memories_from_db(session_world_id)
        except Exception as e:
            print(f"⚠️ Failed to initialize memory system: {e}")
            print("🔄 Continuing without memory enhancement...")
            memory_system = None
    else:
        print("🔇 RAG system disabled by configuration")

    def game_loop(message, history):
        nonlocal last_player_position, number_of_turns, game_log_dictionary
        debug_response = handle_debug_command(message, world, language)
        if debug_response:
            return debug_response

        if not narrate_from_outcome:
            # The single-call pipeline proposes character puzzles here; the new one does it in the engine
            puzzle_response = check_character_puzzle_mention(world, message, language)
            if puzzle_response:
                return puzzle_response

        number_of_turns += 1
        turn_started = time.perf_counter()
        completed_before = objective_completed(world)
        game_log_dictionary[number_of_turns] = {}

        visited_locations.add(world.player.location.name)

        if narrate_from_outcome:
            answer = play_turn_from_outcome(
                world, message, language, reasoning_model, narrative_model,
                number_of_turns, game_log_dictionary, memory_system,
            )
            last_player_position = world.player.location
        else:
            response_update = process_player_input_structured(
                world, message, language, reasoning_model,
                number_of_turns, game_log_dictionary, memory_system,
            )

            step_started = time.perf_counter()
            answer, last_player_position = generate_narration(world, last_player_position, response_update, language, narrative_model)
            print(f"[TIMING] turno: narración de escena {time.perf_counter() - step_started:.1f}s")
        answer = check_objective_completion(world, answer, language, include_completion_text=not narrate_from_outcome,
                                            completed_before=completed_before)

        world_state_formatted = world.format_world_state_for_chat(language=language)
        answer += f"\n\n---\n{world_state_formatted}"

        print(f"\n🌎 World state 🌍\n>Player input: {message}\n{world.render_world(language=language)}\n")
        step_started = time.perf_counter()
        save_game_log(game_log_dictionary, number_of_turns, answer)
        print(f"[TIMING] turno: guardar traza {time.perf_counter() - step_started:.1f}s | total {time.perf_counter() - turn_started:.1f}s")

        return answer.replace("<", r"\<").replace(">", r"\>")

    return game_loop
