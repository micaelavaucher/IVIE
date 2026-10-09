"""Prompts for a game turn in which the engine decides and the narration follows.

1. prompt_interpret_action: the reasoning model turns the player's words into a TurnIntent: what
   the player attempts. It does not narrate and does not decide whether anything works.
2. core/turn_engine.apply_turn checks those attempts against the world's rules.
3. prompt_narrate_outcome: the narrative model writes the scene from the engine's facts.
4. prompt_extract_claims: the reasoning model lists what the narration claims (items received,
   held or offered, passages opened...); core/narration_check.py compares that with the state, so
   a narration that contradicts it is rewritten once before the player reads it.
"""

from .structured_data_models import NarrationClaims, TurnIntent


def prompt_interpret_action(world_state: str, action: str, language: str = 'en', relevant_memories: str = "",
                            characters: list = None, open_puzzles: list = None):
    """characters: every character's name, so an accusation can name someone who is not here.
    open_puzzles: (name, statement) of the puzzles the player can answer here: a puzzle is only
    judged when it is listed in puzzle_attempts."""
    if language == 'es':
        roster = f"Personajes del mundo: {', '.join(characters)}\n\n" if characters else ""
        if open_puzzles:
            roster += "Acertijos abiertos que el jugador puede responder aquí:\n" + "\n".join(
                f'- {name}: "{problem}"' for name, problem in open_puzzles) + "\n\n"
    else:
        roster = f"Characters in the world: {', '.join(characters)}\n\n" if characters else ""
        if open_puzzles:
            roster += "Open puzzles the player can answer here:\n" + "\n".join(
                f'- {name}: "{problem}"' for name, problem in open_puzzles) + "\n\n"
    if language == 'es':
        system_msg = """Eres el intérprete de un juego de ficción interactiva. Tu tarea es entender qué intenta hacer el jugador y expresarlo en los campos del JSON. NO decides si funciona: el motor del juego lo comprueba con sus reglas y otro paso narra el resultado.

CÓMO LLENAR CADA CAMPO (usa siempre los nombres EXACTOS del estado del mundo):
- moved_objects: un objeto que el jugador intenta tomar del lugar, pedirle o aceptarle a un personaje ("new_location": nombre exacto del jugador), darle a un personaje ("new_location": nombre del personaje) o dejar en el lugar ("new_location": nombre de la ubicación actual). Si intenta quitarle algo a un personaje sin pedírselo ni negociar, no lo incluyas.
- location_changed: el lugar al que el jugador intenta ir, si intenta ir a alguno.
- passages_to_open: el pasaje bloqueado que el jugador intenta abrir ("destination": su destino) y, si dice con qué lo abre, ese objeto tal como lo nombra ("using_item").
- examined_items: objetos del estado que el jugador examina, lee, estudia o manipula sin llevárselos.
- items_used: objetos que el jugador usa, empuña o dice tener ("uso mi linterna", "disparo con mi revólver"): el nombre exacto si está en el estado, o como lo nombra el jugador si no está. Inclúyelos aunque no los tenga: el motor lo comprueba.
- characters_addressed: personajes con los que el jugador habla, a los que les pide algo o con los que interactúa.
- items_shown: objetos que el jugador le muestra o presenta a un personaje sin dárselo (pon también a ese personaje en characters_addressed). Si se lo da, va en moved_objects.
- puzzle_attempts: acertijos a los que el jugador intenta responder o que dice haber resuelto ("puzzle_name"), con la respuesta que da copiada tal cual de sus palabras ("answer"), o vacía si no da ninguna ("ya lo resolví"). Inclúyelo SIEMPRE que el jugador proponga una respuesta a un acertijo abierto, aunque no lo nombre ("creo que es el eco"). NO lo incluyas si el jugador solo nombra algo por otro motivo (examinar, pedir o usar un objeto). No juzgues si la respuesta es correcta.
- accusation: si el jugador acusa a alguien o dice quién es el responsable del misterio, el nombre exacto de ese personaje (o el nombre que usó, si no coincide con ninguno). Si no acusa a nadie, déjalo vacío.
- action_summary: una o dos oraciones con lo que hace el jugador y cómo reaccionan los personajes según su descripción y su interacción. No afirmes si las acciones de arriba funcionan.

Incluye todo lo que el jugador intente, aunque creas que no va a funcionar: el motor decide. En particular, si el jugador pide un objeto, inclúyelo SIEMPRE en moved_objects, aunque creas que ya lo tiene o que el personaje no lo va a dar. Si la acción no toca el mundo (una pregunta, mirar alrededor), deja las listas vacías y descríbela en action_summary.

El ESTADO ACTUAL DEL MUNDO manda: quién está en cada lugar y qué lleva cada uno es lo que dice el estado, aunque los recuerdos digan otra cosa.

Tu respuesta debe ser un JSON válido que siga exactamente el modelo TurnIntent. Escribe action_summary en español."""
        user_msg = f"""Estado actual del mundo:
{world_state}

{roster}{relevant_memories}El jugador escribe:
<accion>
{action}
</accion>"""
    else:
        system_msg = """You interpret the player's actions in an interactive fiction game. Your task is to understand what the player attempts and express it in the JSON fields. You do NOT decide whether it works: the game engine checks it against its rules and another step narrates the result.

HOW TO FILL EACH FIELD (always use the EXACT names from the world state):
- moved_objects: an item the player tries to take from the place, ask a character for or accept from them ("new_location": the player's exact name), give to a character ("new_location": the character's name) or leave in the place ("new_location": the current location's name). If they try to take something from a character without asking or bargaining, do not include it.
- location_changed: the place the player tries to go to, if any.
- passages_to_open: the blocked passage the player tries to open ("destination": where it leads) and, if they say what they open it with, that item as they name it ("using_item").
- examined_items: items from the state that the player examines, reads, studies or handles without taking them.
- items_used: items the player uses, wields or says they carry ("I use my lantern", "I shoot with my revolver"): the exact name if it is in the state, or as the player names it if not. Include them even if the player does not have them: the engine checks.
- characters_addressed: characters the player talks to, asks something or interacts with.
- items_shown: items the player shows or presents to a character without giving them (also list that character in characters_addressed). If they give it, it goes in moved_objects.
- puzzle_attempts: puzzles the player tries to answer or says they solved ("puzzle_name"), with the answer they give copied word for word from what they wrote ("answer"), or empty if they give none ("I solved it"). ALWAYS include it when the player proposes an answer to an open puzzle, even without naming it ("I think it's an echo"). Do NOT include it when the player only names something for another reason (examining, asking for or using an item). Do not judge whether the answer is right.
- accusation: if the player accuses someone or says who is responsible for the mystery, that character's exact name (or the name they used, if it matches none). Leave it empty if they accuse nobody.
- action_summary: one or two sentences with what the player does and how the characters react, according to their descriptions and interactions. Do not state whether the attempts above succeed.

Include everything the player attempts, even if you think it will not work: the engine decides. In particular, if the player asks for an item, ALWAYS include it in moved_objects, even if you think they already have it or the character will not give it. If the action does not touch the world (a question, looking around), leave the lists empty and describe it in action_summary.

The CURRENT WORLD STATE wins: who is in each place and what each one carries is what the state says, even if the memories say otherwise.

Your answer must be valid JSON following exactly the TurnIntent model. Write action_summary in English."""
        user_msg = f"""Current world state:
{world_state}

{roster}{relevant_memories}The player writes:
<action>
{action}
</action>"""
    return system_msg, user_msg, TurnIntent


def prompt_narrate_outcome(world_state: str, action: str, action_summary: str, facts: str,
                           previous_narrations: list, arrived: bool, first_visit: bool,
                           objective: str, objective_completed: bool, language: str = 'en',
                           completed_before: bool = False, corrections: list = None):
    """objective_completed: the objective was completed in this turn, so the narration closes the
    adventure. completed_before: it was completed in an earlier turn; the ending is not rewritten.
    corrections: what a first version of this narration got wrong (core/narration_check.py)."""
    previous = "\n\n".join(previous_narrations[-2:]) if previous_narrations else ""
    if corrections and language == 'es':
        correction_block = ("\n\nUNA VERSIÓN ANTERIOR DE ESTA NARRACIÓN CONTRADECÍA EL JUEGO. Escríbela de nuevo sin esto:\n"
                            + "\n".join(f"- {c}" for c in corrections))
    elif corrections:
        correction_block = ("\n\nA PREVIOUS VERSION OF THIS NARRATION CONTRADICTED THE GAME. Write it again without this:\n"
                            + "\n".join(f"- {c}" for c in corrections))
    else:
        correction_block = ""
    if language == 'es':
        objective_rule = (("El jugador ACABA DE CUMPLIR el objetivo: cierra la aventura con un final breve y satisfactorio, "
                          "basado solo en lo que ocurrió y en los elementos del mundo, sin acciones que el jugador no hizo ni objetos que no tiene.")
                          if objective_completed else
                          "El objetivo ya se cumplió en un turno anterior y la aventura terminó: narra solo esta acción, sin volver a escribir el final."
                          if completed_before else
                          "El objetivo NO está cumplido: no lo celebres ni digas que la misión terminó o que el misterio está resuelto.")
        arrival = ("El jugador acaba de llegar a este lugar por primera vez: descríbelo con detalle." if arrived and first_visit
                   else "El jugador acaba de volver a este lugar: descríbelo brevemente." if arrived else "")
        system_msg = f"""Eres el narrador de un juego de ficción interactiva. Escribe en segunda persona lo que ocurre como resultado de la acción del jugador.

REGLAS ESTRICTAS:
1. Los HECHOS DEL MOTOR son la única verdad sobre lo que ocurrió. Narra cada uno; lo marcado con ✗ es un intento que no funcionó: cuéntalo así, con su motivo.
2. No cuentes ningún cambio que no esté en los hechos: nadie entrega, toma, abre, resuelve ni descubre nada que no figure ahí. Examinar un objeto no es llevárselo. Resolver un acertijo no revela nada más que lo que dicen los hechos. No inventes objetos, personajes, pistas, marcas ni mensajes que no estén en el estado del mundo: puedes adornar lo que existe, no agregar cosas nuevas.
2b. El jugador tiene EXACTAMENTE los objetos de su inventario en el estado del mundo. Si menciona usar algo que no tiene, cuenta que no lo tiene; nunca lo describas usándolo. No digas que dio una respuesta si los hechos dicen que no dio ninguna.
2c. Los hechos ocurren ahora, en este turno: lo que el jugador recibe o el lugar al que llega en este turno es nuevo; nunca digas que ya lo tenía o que ya estaba ahí.
3. Los personajes presentes son exactamente los del estado del mundo: no digas que alguien está o falta si eso contradice el estado.
4. Si se presenta un acertijo, incluye su enunciado tal cual. Nunca reveles ni insinúes la respuesta de un acertijo.
5. {objective_rule}
6. Usa las descripciones del mundo para darle vida a la escena, con detalles sensoriales, pero sé conciso: uno o dos párrafos. {arrival}
7. Si el jugador no cambió de lugar, no vuelvas a describir el lugar: cuenta lo que pasó con su acción. No repitas frases de las narraciones anteriores de este lugar.
8. Responde solo con la narración en prosa, sin títulos, listas, comentarios ni las marcas ✓/✗ de los hechos. Escribe en español."""
        user_msg = f"""Estado del mundo después de la acción:
{world_state}

Narraciones anteriores de este lugar (no las repitas):
{previous or '(ninguna)'}

Objetivo del jugador: {objective or '(sin objetivo)'}

Acción del jugador: {action}
Intención interpretada (contexto, NO son hechos): {action_summary}

HECHOS DEL MOTOR:
{facts}{correction_block}"""
    else:
        objective_rule = (("The player HAS JUST COMPLETED the objective: close the adventure with a short, satisfying ending, "
                          "based only on what happened and on the elements of the world, without actions the player did not take or items they do not have.")
                          if objective_completed else
                          "The objective was completed in an earlier turn and the adventure is over: narrate only this action, without writing the ending again."
                          if completed_before else
                          "The objective is NOT completed: do not celebrate it or say the mission is over or the mystery is solved.")
        arrival = ("The player has just arrived here for the first time: describe the place in detail." if arrived and first_visit
                   else "The player has just come back here: describe the place briefly." if arrived else "")
        system_msg = f"""You are the narrator of an interactive fiction game. Write, in the second person, what happens as a result of the player's action.

STRICT RULES:
1. The ENGINE FACTS are the only truth about what happened. Narrate each one; those marked ✗ are attempts that did not work: tell them as such, with their reason.
2. Do not tell any change that is not in the facts: nobody hands over, takes, opens, solves or discovers anything that is not there. Examining an item is not taking it. Solving a puzzle reveals nothing beyond what the facts say. Do not invent items, characters, clues, marks or messages that are not in the world state: you may embellish what exists, not add new things.
2b. The player has EXACTLY the items in their inventory in the world state. If they mention using something they do not have, tell them they do not have it; never describe them using it. Do not say they gave an answer if the facts say they gave none.
2c. The facts happen now, in this turn: what the player receives or the place they reach this turn is new; never say they already had it or were already there.
3. The characters present are exactly those in the world state: do not say someone is here or missing if that contradicts the state.
4. If a puzzle is presented, include its statement verbatim. Never reveal or hint at a puzzle's answer.
5. {objective_rule}
6. Use the world's descriptions to bring the scene to life with sensory detail, but be concise: one or two paragraphs. {arrival}
7. If the player did not change places, do not describe the place again: tell what happened with their action. Do not repeat sentences from the previous narrations of this place.
8. Answer only with the narration, in prose, without titles, lists, comments or the facts' ✓/✗ marks. Write in English."""
        user_msg = f"""World state after the action:
{world_state}

Previous narrations of this place (do not repeat them):
{previous or '(none)'}

The player's objective: {objective or '(no objective)'}

Player's action: {action}
Interpreted intent (context, NOT facts): {action_summary}

ENGINE FACTS:
{facts}{correction_block}"""
    return system_msg, user_msg


def prompt_extract_claims(narration: str, item_names: list, language: str = 'en'):
    """Ask what the narration claims about items, passages, answers and the objective. The model
    does not judge the claims: the engine checks them against the state."""
    names = ", ".join(item_names) or "-"
    if language == 'es':
        system_msg = """Lees la narración de un turno de un juego de ficción interactiva y extraes, sin juzgarlo, lo que AFIRMA que pasa. Llena los campos del JSON NarrationClaims solo con lo que la narración dice explícitamente. Usa los nombres exactos de la lista de objetos cuando el objeto está en ella; si no, escríbelo como lo nombra la narración. Lo que la narración niega ("no tienes linterna") no se incluye. Los muebles y el decorado del lugar no son objetos que se reciben, tienen u ofrecen."""
        user_msg = f"""Objetos del mundo: {names}

Narración:
{narration}"""
    else:
        system_msg = """You read the narration of one turn of an interactive fiction game and extract, without judging it, what it CLAIMS happens. Fill the fields of the NarrationClaims JSON only with what the narration states explicitly. Use the exact names from the item list when the item is in it; otherwise write it as the narration names it. What the narration denies ("you have no lantern") is not included. Furniture and scenery are not items that are received, held or offered."""
        user_msg = f"""Items in the world: {names}

Narration:
{narration}"""
    return system_msg, user_msg, NarrationClaims
