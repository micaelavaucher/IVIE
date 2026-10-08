"""Prompts for the facet stage (step 3) and the puzzle stage (step 4) of world generation.

Adapted from the ivie-msc implementation (Micaela Vaucher's master's thesis repository).

The original step 3 asked one call to emit an entire GeneratedWorld. These prompts split that
single ask into four independent ones - locations, items, characters, objective - each
emitting only its own slice of the world. The rules that used to live in one giant step-3
prompt are redistributed here, each to the facet it actually governs.

Every facet receives the SAME roster of names (from the skeleton). That shared roster is
what lets the four calls run without seeing each other's output: a facet may reference any
name on the roster, and reconcile_world_facets settles the places where two of them
disagree about the same item.

Every prompt takes `previous_errors`: when a retry follows a rejected attempt, the reasons are
handed back so the model corrects them instead of repeating the same mistake.
"""

#---- Shared fragments --------------------------------------------------------

def _language_instruction(language: str) -> str:
    if language == 'es':
        return (
            "\n**INSTRUCCIÓN DE IDIOMA CRÍTICA: La respuesta DEBE estar íntegramente en español. "
            "Todos los valores de texto (nombres, descripciones, etc.) deben ser generados en español. "
            "Las claves del JSON deben permanecer en inglés para coincidir con el esquema solicitado.**\n"
        )
    return (
        "\n**CRITICAL LANGUAGE INSTRUCTION: The response MUST be entirely in English. "
        "All text values (names, descriptions, etc.) must be generated in English. "
        "The JSON keys must remain in English to match the required schema.**\n"
    )


def _feedback_block(previous_errors, language: str) -> str:
    if not previous_errors:
        return ""
    listed = "\n".join(f"- {error}" for error in previous_errors)
    header = ("**TU INTENTO ANTERIOR FUE RECHAZADO POR ESTOS ERRORES - CORRÍGELOS:**" if language == 'es'
              else "**YOUR PREVIOUS ATTEMPT WAS REJECTED FOR THESE ERRORS - FIX THEM:**")
    return f"\n{header}\n{listed}\n"


def _concept_block(concept, language: str) -> str:
    if language == 'es':
        return f"""Concepto del mundo:
- Título: {concept.title}
- Historia de fondo: {concept.backstory}
- Concepto del jugador: {concept.player_concept}
- Objetivo principal: {concept.main_objective}"""
    return f"""World concept:
- Title: {concept.title}
- Backstory: {concept.backstory}
- Player concept: {concept.player_concept}
- Main objective: {concept.main_objective}"""


def _roster_block(skeleton, language: str) -> str:
    """The complete name roster, handed identically to every facet.

    This is the contract that makes the facets independent: each one may reference any name
    listed here, even though it never sees the other facets' output.
    """
    locations = "\n".join(f"- {entity.name}: {entity.purpose}" for entity in skeleton.key_locations)
    items = "\n".join(f"- {entity.name}: {entity.purpose}" for entity in skeleton.key_items)
    characters = "\n".join(f"- {entity.name}: {entity.purpose}" for entity in skeleton.key_characters)

    if language == 'es':
        return f"""ELEMENTOS DEL MUNDO (nombres fijos - úsalos EXACTAMENTE así, no inventes otros):

Ubicaciones:
{locations}

Objetos:
{items}

Personajes:
{characters}"""
    return f"""WORLD ELEMENTS (fixed names - use them EXACTLY as written, do not invent others):

Locations:
{locations}

Items:
{items}

Characters:
{characters}"""


#---- Locations facet --------------------------------------------------------

def PROMPT_FACET_LOCATIONS(concept, skeleton, language: str = 'es', previous_errors=None) -> str:
    """Ask for location descriptions, topology, item placement and blocked passages."""
    if language == 'es':
        return f"""Eres un constructor de mundos de ficción interactiva. Tu ÚNICA tarea en este paso es desarrollar las UBICACIONES del mundo: sus descripciones, cómo se conectan entre sí, qué objetos yacen en ellas y qué pasajes están bloqueados.

{_concept_block(concept, language)}

{_roster_block(skeleton, language)}
{_feedback_block(previous_errors, language)}
**GENERA SOLO LAS UBICACIONES.** No generes objetos, personajes ni el objetivo: otros pasos se encargan de eso. Solo puedes REFERENCIAR objetos por su nombre exacto de la lista de arriba.

**RESTRICCIÓN DE TAMAÑO CRÍTICA:**
- SOLO puedes usar las ubicaciones listadas arriba - PROHIBIDO crear nuevas ubicaciones
- Debes desarrollar TODAS las ubicaciones listadas, ni una más ni una menos

**REGLAS DE CONEXIÓN OBLIGATORIAS:**
1. **Conexiones bidireccionales**: Si A conecta con B, entonces B DEBE conectar con A
2. **Conectividad global**: TODAS las ubicaciones deben ser accesibles - NO puede haber ubicaciones aisladas ni grupos separados
3. **Un pasaje bloqueado sigue siendo una conexión**: debe aparecer también en `connecting_locations`

**REGLAS DE COLOCACIÓN DE OBJETOS:**
1. En `items` de cada ubicación, pon los nombres de los objetos que se encuentran ahí, tomados EXACTAMENTE de la lista de objetos
2. **REGLA LÓGICA CRÍTICA**: Un objeto necesario para desbloquear un pasaje NO PUEDE estar en la ubicación que está detrás de ese mismo pasaje, ni en ninguna ubicación accesible únicamente a través de él
3. No todos los objetos tienen que estar en ubicaciones: algunos los llevarán los personajes. Deja fuera los que tenga más sentido que lleve un personaje
4. Nunca pongas el mismo objeto en dos ubicaciones distintas

**REGLAS DE PASAJES BLOQUEADOS:**
1. `obstacle_name` es el objeto físico que bloquea (ej. "Candado", "Puerta atrancada"), y debe ser DISTINTO de lo que se necesita para quitarlo
2. `required_to_unblock` DEBE ser un requisito de tipo `item` que nombre un objeto de la lista. El motor solo sabe abrir un pasaje con un OBJETO (o, más adelante, con un puzzle). Un requisito de personaje o de ubicación se descarta en silencio y el pasaje queda abierto, así que no los uses
3. Los pasajes bloqueados deben mantener el mundo completable: nunca encierres detrás de un bloqueo el objeto o personaje que hace falta para superarlo
{_language_instruction(language)}"""

    return f"""You are a builder of interactive fiction worlds. Your ONLY task in this step is to develop the world's LOCATIONS: their descriptions, how they connect, which items lie in them, and which passages are blocked.

{_concept_block(concept, language)}

{_roster_block(skeleton, language)}
{_feedback_block(previous_errors, language)}
**GENERATE ONLY THE LOCATIONS.** Do not generate items, characters, or the objective - other steps handle those. You may only REFERENCE items by their exact name from the list above.

**CRITICAL SIZE CONSTRAINT:**
- You may ONLY use the locations listed above - creating new locations is FORBIDDEN
- You must develop ALL the listed locations, no more and no fewer

**MANDATORY CONNECTION RULES:**
1. **Bidirectional connections**: If A connects to B, then B MUST connect to A
2. **Global connectivity**: ALL locations must be reachable - there can be no isolated locations or separate clusters
3. **A blocked passage is still a connection**: it must also appear in `connecting_locations`

**ITEM PLACEMENT RULES:**
1. In each location's `items`, list the names of items found there, taken EXACTLY from the item list
2. **CRITICAL LOGIC RULE**: An item needed to unlock a passage CANNOT be placed in the location behind that very passage, nor in any location reachable only through it
3. Not every item needs to be in a location - some will be carried by characters. Leave out the ones it makes more sense for a character to hold
4. Never place the same item in two different locations

**BLOCKED PASSAGE RULES:**
1. `obstacle_name` is the physical object doing the blocking (e.g. "Padlock", "Barred door"), and must be DIFFERENT from what is needed to remove it
2. `required_to_unblock` MUST be an `item` requirement naming an item from the list. The engine can only open a passage with an ITEM (or, later, a puzzle). A character or location requirement is silently discarded and the passage is left open, so do not use them
3. Blocked passages must keep the world completable: never seal the item or character needed to get past a block behind that same block
{_language_instruction(language)}"""


#---- Items facet ------------------------------------------------------------

def PROMPT_FACET_ITEMS(concept, skeleton, language: str = 'es', previous_errors=None) -> str:
    """Ask for item detail only - placement is decided by the locations and characters facets."""
    if language == 'es':
        return f"""Eres un constructor de mundos de ficción interactiva. Tu ÚNICA tarea en este paso es desarrollar los OBJETOS del mundo: qué son, cómo se describen y para qué sirven mecánicamente.

{_concept_block(concept, language)}

{_roster_block(skeleton, language)}
{_feedback_block(previous_errors, language)}
**GENERA SOLO LOS OBJETOS.** No generes ubicaciones, personajes ni el objetivo. NO decidas dónde está cada objeto: la colocación la deciden otros pasos.

**RESTRICCIÓN DE TAMAÑO CRÍTICA:**
- Desarrolla EXACTAMENTE los objetos listados arriba, ni uno más ni uno menos
- Usa sus nombres EXACTAMENTE como aparecen

**REGLAS DE OBJETOS:**
1. **action_type**: elige la ÚNICA acción mecánica que el motor de juego soporta para ese objeto:
   - `unlock_passage`: el objeto es la llave de un pasaje bloqueado
   - `solve_puzzle`: el objeto es clave o pista para resolver un puzzle
   - `give_to_character`: el objeto está pensado para entregárselo a un personaje
   - `lore`: el objeto aporta historia o información, sin uso mecánico
2. **Objetivos completables**: si un objeto es requerido para el objetivo (`is_objective_target: true`), DEBE ser `gettable: true`
3. **Consistencia funcional**: los objetos decorativos pueden ser `gettable: false`; los objetos funcionales DEBEN ser `gettable: true`
4. **Relevancia clara**: cada objeto debe tener una razón de existir, funcional o atmosférica. Rellena `relevance_to_objective` cuando sea funcional
5. **required_for**: nombra los puzzles, personajes o ubicaciones para los que hace falta este objeto, usando nombres de la lista
{_language_instruction(language)}"""

    return f"""You are a builder of interactive fiction worlds. Your ONLY task in this step is to develop the world's ITEMS: what they are, how they are described, and what they mechanically do.

{_concept_block(concept, language)}

{_roster_block(skeleton, language)}
{_feedback_block(previous_errors, language)}
**GENERATE ONLY THE ITEMS.** Do not generate locations, characters, or the objective. Do NOT decide where each item sits - placement is decided by other steps.

**CRITICAL SIZE CONSTRAINT:**
- Develop EXACTLY the items listed above, no more and no fewer
- Use their names EXACTLY as written

**ITEM RULES:**
1. **action_type**: pick the ONE mechanical action the game engine supports for that item:
   - `unlock_passage`: the item is the key to a blocked passage
   - `solve_puzzle`: the item is a key or clue for solving a puzzle
   - `give_to_character`: the item is meant to be handed to a character
   - `lore`: the item carries story or information, with no mechanical use
2. **Completable objectives**: if an item is required for the objective (`is_objective_target: true`), it MUST be `gettable: true`
3. **Functional consistency**: decorative items may be `gettable: false`; functional items MUST be `gettable: true`
4. **Clear relevance**: every item needs a reason to exist, functional or atmospheric. Fill in `relevance_to_objective` when it is functional
5. **required_for**: name the puzzles, characters, or locations this item is needed for, using names from the list
{_language_instruction(language)}"""


#---- Characters facet -------------------------------------------------------

def PROMPT_FACET_CHARACTERS(concept, skeleton, language: str = 'es', previous_errors=None) -> str:
    """Ask for the cast: NPCs and the player."""
    if language == 'es':
        return f"""Eres un constructor de mundos de ficción interactiva. Tu ÚNICA tarea en este paso es desarrollar los PERSONAJES del mundo: los NPCs y el personaje jugador.

{_concept_block(concept, language)}

{_roster_block(skeleton, language)}
{_feedback_block(previous_errors, language)}
**GENERA SOLO LOS PERSONAJES.** No generes ubicaciones, objetos ni el objetivo. Solo puedes REFERENCIAR ubicaciones y objetos por su nombre exacto de las listas de arriba.

**RESTRICCIÓN DE TAMAÑO CRÍTICA:**
- Desarrolla EXACTAMENTE los personajes listados arriba como NPCs, ni uno más ni uno menos
- Además, genera el `player`: el personaje jugador, que NO está en la lista de personajes

**REGLAS DE PERSONAJES:**
1. **Ubicaciones válidas**: el `location` de cada personaje DEBE ser el nombre exacto de una ubicación de la lista
2. **Inventarios válidos**: todo objeto en `inventory` DEBE ser el nombre exacto de un objeto de la lista
3. **Interacciones funcionales**: si un personaje tiene `interaction`, DEBE tener `interaction_text`
4. **REGLA LÓGICA CRÍTICA**: si un personaje lleva un objeto o tiene la solución necesaria para desbloquear un pasaje, NO PUEDE estar en la ubicación que está detrás de ese mismo pasaje, ni en ninguna accesible únicamente a través de él
5. **El jugador**: colócalo en la ubicación donde empieza la aventura, con el inventario inicial VACÍO: los objetos se consiguen explorando e interactuando

IMPORTANTE: las interacciones deben ser directas y simples pero COMPLETAS. Todavía no hay puzzles: no rellenes `proposes_puzzle`, eso se añadirá en un paso posterior.
{_language_instruction(language)}"""

    return f"""You are a builder of interactive fiction worlds. Your ONLY task in this step is to develop the world's CHARACTERS: the NPCs and the player character.

{_concept_block(concept, language)}

{_roster_block(skeleton, language)}
{_feedback_block(previous_errors, language)}
**GENERATE ONLY THE CHARACTERS.** Do not generate locations, items, or the objective. You may only REFERENCE locations and items by their exact names from the lists above.

**CRITICAL SIZE CONSTRAINT:**
- Develop EXACTLY the characters listed above as NPCs, no more and no fewer
- In addition, generate the `player`: the player character, who is NOT in the character list

**CHARACTER RULES:**
1. **Valid locations**: each character's `location` MUST be the exact name of a location from the list
2. **Valid inventories**: every item in `inventory` MUST be the exact name of an item from the list
3. **Functional interactions**: if a character has an `interaction`, it MUST have `interaction_text`
4. **CRITICAL LOGIC RULE**: if a character holds an item or the solution needed to unlock a passage, they CANNOT be placed in the location behind that very passage, nor in any location reachable only through it
5. **The player**: place them where the adventure begins, with an EMPTY starting inventory: items are obtained by exploring and interacting

IMPORTANT: interactions should be direct and simple but COMPLETE. There are no puzzles yet: do not fill in `proposes_puzzle`, that is added in a later step.
{_language_instruction(language)}"""


#---- Objective facet --------------------------------------------------------

def PROMPT_FACET_OBJECTIVE(concept, skeleton, language: str = 'es', previous_errors=None) -> str:
    """Ask for the main objective plus the framing prose (theme, narrative context)."""
    feedback = _feedback_block(previous_errors, language)

    if language == 'es':
        return f"""Eres un diseñador de aventuras de ficción interactiva. Tu ÚNICA tarea en este paso es definir el OBJETIVO PRINCIPAL del mundo, junto con su tema y su contexto narrativo.

{_concept_block(concept, language)}

{_roster_block(skeleton, language)}
{feedback}
**GENERA SOLO EL OBJETIVO, EL TEMA Y EL CONTEXTO NARRATIVO.** No generes ubicaciones, objetos ni personajes: solo puedes REFERENCIARLOS por su nombre exacto.

**INSTRUCCIONES CRÍTICAS PARA LOS COMPONENTES DEL OBJETIVO:**
- El objetivo DEBE tener SOLO UN COMPONENTE PRINCIPAL que representa la meta final
- Para objetivos tipo "reach_location": UN solo componente de tipo "location" (la ubicación de destino)
- Para objetivos tipo "get_item": UN solo componente de tipo "item" (el objeto que obtener)
- Para objetivos tipo "find_character": UN solo componente de tipo "character" (el personaje que encontrar)
- Para objetivos tipo "deliver_an_item": EXACTAMENTE DOS componentes - UNO de tipo "item" (el objeto a entregar) y UNO de tipo "character" o "location" (el destinatario). Si solo pones el destino y olvidas el item, el objetivo es imposible de completar y será rechazado
- NO incluyas elementos auxiliares o herramientas como componentes principales
- Los componentes principales son SOLO la meta final, no los medios para alcanzarla
- Todo componente DEBE existir con su nombre exacto en las listas de arriba
- La `description` del objetivo debe corresponder EXACTAMENTE a sus componentes: si el componente es UN objeto, no hables de reunir varios; no menciones elementos que no estén en las listas, ni una cantidad de pistas distinta de las `mystery_clues` que defines

**SI EL OBJETIVO ES UN MISTERIO (type: "solve_mystery"), DEBES incluir:**
1. **mystery_clues**: al menos 3-5 pistas que el jugador pueda descubrir, que conduzcan lógicamente a la solución
2. **mystery_solution**: la solución completa del misterio
3. **CRÍTICO**: `associated_item` DEBE ser el nombre exacto de un OBJETO de la lista - NUNCA un personaje ni una ubicación. Los jugadores descubren pistas al interactuar con objetos físicos
3b. **CRÍTICO - UN OBJETO DISTINTO POR PISTA**: dos pistas NO pueden compartir el mismo `associated_item`. Cada pista debe apuntar a un objeto diferente, porque un objeto solo se consigue una vez. Si tienes menos objetos que pistas, escribe menos pistas
4. **CRÍTICO**: `item_location` DEBE ser el nombre exacto de una UBICACIÓN de la lista

**SI EL OBJETIVO NO ES UN MISTERIO, DEBES incluir:**
1. **completion_narration**: una descripción narrativa de lo que sucede tras completar el objetivo, que dé una conclusión satisfactoria y explique las consecuencias del logro
2. **objective_hints**: EXACTAMENTE 3 pistas progresivas (de general a específica), que se usarán cuando el jugador esté en una ubicación sin puzzles y pida ayuda

Ejemplos de completion_narration:
- "Con la espada legendaria en tus manos, finalmente derrotas al dragón. El reino está a salvo y eres aclamado como un héroe."
- "Al entregar todos los documentos requeridos, el oficial te otorga el permiso. Ahora puedes continuar tu viaje hacia tierras lejanas."

**VALIDACIÓN:** antes de finalizar, verifica que el objetivo sea alcanzable con los elementos listados, y que `success_conditions` describa condiciones realmente comprobables.
{_language_instruction(language)}"""

    return f"""You are a designer of interactive fiction adventures. Your ONLY task in this step is to define the world's MAIN OBJECTIVE, along with its theme and narrative context.

{_concept_block(concept, language)}

{_roster_block(skeleton, language)}
{feedback}
**GENERATE ONLY THE OBJECTIVE, THE THEME, AND THE NARRATIVE CONTEXT.** Do not generate locations, items, or characters - you may only REFERENCE them by their exact names.

**CRITICAL INSTRUCTIONS FOR OBJECTIVE COMPONENTS:**
- The objective MUST have only ONE PRINCIPAL COMPONENT representing the final goal
- For "reach_location" objectives: a single component of type "location" (the destination)
- For "get_item" objectives: a single component of type "item" (the item to obtain)
- For "find_character" objectives: a single component of type "character" (the character to find)
- For "deliver_an_item" objectives: EXACTLY TWO components - ONE of type "item" (the thing delivered) and ONE of type "character" or "location" (the recipient). If you give only the destination and omit the item, the objective is impossible to complete and will be rejected
- Do NOT include auxiliary elements or tools as principal components
- Principal components are ONLY the final goal, not the means of reaching it
- Every component MUST exist by its exact name in the lists above
- The objective's `description` must match its components EXACTLY: if the component is ONE item, do not talk about gathering several; do not mention elements that are not in the lists, nor a number of clues different from the `mystery_clues` you define

**IF THE OBJECTIVE IS A MYSTERY (type: "solve_mystery"), you MUST include:**
1. **mystery_clues**: at least 3-5 clues the player can discover, leading logically to the solution
2. **mystery_solution**: the complete solution to the mystery
3. **CRITICAL**: `associated_item` MUST be the exact name of an ITEM from the list - NEVER a character or a location. Players discover clues by interacting with physical objects
3b. **CRITICAL - ONE DISTINCT ITEM PER CLUE**: two clues may NOT share the same `associated_item`. Each clue must point at a different item, because an item is only ever acquired once. If you have fewer items than clues, write fewer clues
4. **CRITICAL**: `item_location` MUST be the exact name of a LOCATION from the list

**IF THE OBJECTIVE IS NOT A MYSTERY, you MUST include:**
1. **completion_narration**: a narrative description of what happens after the objective is completed, giving a satisfying conclusion and explaining the consequences of the achievement
2. **objective_hints**: EXACTLY 3 progressive hints (general to specific), used when the player is in a location with no puzzles and asks for help

Examples of completion_narration:
- "With the legendary sword in your hands, you finally defeat the dragon. The kingdom is safe and you are hailed as a hero."
- "Handing over all the required documents, the official grants you the permit. Now you can continue your journey to distant lands."

**VALIDATION:** before finalizing, check that the objective is achievable with the listed elements, and that `success_conditions` describes genuinely checkable conditions.
{_language_instruction(language)}"""


#---- Puzzle stage -----------------------------------------------------------

def PROMPT_PUZZLE_STAGE(world_data: str, puzzle_count: str, language: str = 'es', previous_errors=None) -> str:
    """Ask for puzzles (and the passages they gate) against an already-finished world.

    The model still sees the whole world, because good puzzle chains need that context. What
    changed is the OUTPUT contract: it returns only puzzles and blocked-passage additions, so
    it can no longer redistribute items, invent locations, or rewrite characters - the three
    things the old whole-world step did that the pipeline then had to detect and retry.
    """
    feedback = _feedback_block(previous_errors, language)

    if language == 'es':
        return f"""Eres un diseñador de puzzles para ficción interactiva. Tu tarea es añadir puzzles y obstáculos al mundo existente para hacer la aventura más desafiante e interesante.

Mundo actual:
{world_data}
{feedback}
**QUÉ DEBES DEVOLVER - LÉELO PRIMERO:**
Devuelves SOLO dos cosas:
1. `puzzles`: EXACTAMENTE {puzzle_count} puzzle(s) nuevo(s)
2. `blocked_passage_additions`: los pasajes que haya que bloquear para que esos puzzles tengan algo que desbloquear

NO devuelves ubicaciones, objetos, personajes ni el objetivo. Ya están decididos y NO se pueden cambiar:
- PROHIBIDO crear ubicaciones, objetos o personajes nuevos
- PROHIBIDO mover objetos de sitio o cambiar inventarios
- PROHIBIDO reescribir descripciones o interacciones existentes
- Todo lo que cites DEBE existir ya en el mundo de arriba, escrito EXACTAMENTE igual

No hace falta que indiques qué personaje propone cada puzzle en la ficha del personaje: basta con
poner `proposed_by_character` en el puzzle, y el motor lo conecta solo.

**OBJETIVO CRÍTICO:** En lugar de simplemente añadir puzzles aislados, los puzzles deben formar parte de CADENAS DE DEPENDENCIAS que hagan la progresión hacia el objetivo más interesante y desafiante.

**EJEMPLO DE CADENA DE DEPENDENCIAS:**
Objetivo: Conseguir el Amuleto Mágico
1. El Amuleto está en la Habitación Secreta, pero la puerta está bloqueada por un Candado Mágico
2. Para abrir el Candado necesitas la Palabra Clave que solo conoce el Mago
3. El Mago te la dará, pero primero quiere que le traigas su Bastón Perdido
4. El Bastón lo tiene el Comerciante, que lo intercambiará por las Gemas
5. Las Gemas se consiguen resolviendo el Acertijo del Guardián

**ESTRUCTURA DE CADENAS:**
1. **Cadena principal**: la ruta hacia el objetivo debe tener varios pasos interdependientes, tantos como permita el tamaño del mundo
2. **Progresión lógica**: cada paso debe ser consecuencia lógica del anterior
3. **Motivación clara**: cada personaje debe tener una razón convincente para lo que pide
4. **Conectividad temática**: todas las tareas deben conectar con la historia principal

**REGLAS TÉCNICAS OBLIGATORIAS:**
1. **Conectividad global**: los pasajes que bloquees NO pueden dejar ubicaciones inalcanzables. Nunca bloquees el único camino hacia algo que hace falta para desbloquearlo
2. **Soluciones descubribles**: cada puzzle DEBE tener pistas que el jugador pueda encontrar explorando, sin conocimiento externo
3. **Obstáculos lógicos**: el obstáculo debe estar separado de su solución
4. **Si el puzzle lo propone un personaje**, la recompensa debe ser un objeto que YA ESTÉ EN EL INVENTARIO DE ESE PERSONAJE
5. **NO CRAFTING**: el motor no soporta crear ni transformar objetos. No pidas al jugador combinar o alterar objetos. Las recompensas son objetos que ya existen y se usan directamente
6. **OBSTÁCULO vs. LLAVE**: el `obstacle_name` es lo que bloquea el camino (p. ej. "puerta cerrada"); el `required_to_unblock` es la llave que lo quita (p. ej. "llave de hierro"). La llave nunca es el obstáculo. `required_to_unblock` solo puede ser de tipo `item` (un objeto que exista) o `puzzle` (un puzzle que estés creando): el motor no sabe abrir un pasaje de ninguna otra forma
7. **QUIÉN PROPONE EL PUZZLE**: un puzzle lo propone un personaje (`proposed_by_character`) o un OBJETO que el jugador examina. Para lo segundo usa `proposed_by_location` con el nombre EXACTO de ese OBJETO: a pesar del nombre del campo, NO es una ubicación. La ubicación del puzzle va en `location`
8. **RESPUESTA CORTA**: `answer` debe ser una palabra o una frase muy breve que el jugador pueda escribir tal cual (ej. "luna", "1885", "el reloj"). El motor busca esa respuesta en lo que escribe el jugador, así que una explicación larga hace el puzzle imposible. En `accepted_answers` pon de 2 a 5 formas cortas en que un jugador podría decir la MISMA respuesta (la palabra clave sola, sin artículo, sinónimos evidentes). Ej.: answer "El Eco Silente" → accepted_answers ["eco", "el eco", "eco silente"]
9. **RECOMPENSAS NUEVAS**: nunca recompenses con un objeto que el jugador ya lleve en su inventario al empezar, ni bloquees un pasaje con un objeto que ya lleva: lo tendría gratis

**REGLAS OBLIGATORIAS PARA PISTAS:**
Para CADA puzzle DEBES incluir:
1. **puzzle_hints**: 3-5 pistas progresivas (de general a específica)
2. **interaction_hint**: cómo interactuar con el puzzle si el jugador no ha empezado
   - Puzzles de personaje: "Intenta hablar con [personaje]"
   - Puzzles de objeto: "Intenta examinar [objeto]"
   - Puzzles de ubicación: "Busca pistas en esta ubicación"

**VALIDACIÓN:** antes de finalizar, verifica mentalmente:
1. ¿Cada puzzle es resoluble con lo que ya existe en el mundo?
2. ¿Todas las ubicaciones siguen siendo alcanzables tras tus bloqueos?
3. ¿Todos los elementos citados existen y están escritos exactamente igual?
4. ¿Los puzzles son interesantes pero no frustrantes, y encajan con la temática?
{_language_instruction(language)}"""

    return f"""You are a puzzle designer for interactive fiction. Your task is to add puzzles and obstacles to the existing world to make the adventure more challenging and interesting.

Current world:
{world_data}
{feedback}
**WHAT YOU MUST RETURN - READ THIS FIRST:**
You return ONLY two things:
1. `puzzles`: EXACTLY {puzzle_count} new puzzle(s)
2. `blocked_passage_additions`: the passages to newly block, so those puzzles have something to unlock

You do NOT return locations, items, characters, or the objective. Those are already decided and CANNOT be changed:
- Creating new locations, items, or characters is FORBIDDEN
- Moving items around or changing inventories is FORBIDDEN
- Rewriting existing descriptions or interactions is FORBIDDEN
- Everything you reference MUST already exist in the world above, spelled EXACTLY the same

You do not need to state which character proposes a puzzle on the character's own record: just set
`proposed_by_character` on the puzzle, and the engine wires it up.

**CRITICAL GOAL:** Rather than adding isolated puzzles, the puzzles must be part of DEPENDENCY CHAINS that make progression toward the objective more interesting and challenging.

**EXAMPLE DEPENDENCY CHAIN:**
Objective: Obtain the Magic Amulet
1. The Amulet is in the Secret Room, but the door is blocked by a Magic Padlock
2. To open the Padlock you need the Password, known only to the Wizard
3. The Wizard will tell you, but first wants his Lost Staff brought back
4. The Staff is held by the Merchant, who will trade it for the Gems
5. The Gems come from solving the Guardian's Riddle

**CHAIN STRUCTURE:**
1. **Main chain**: the route to the objective should have several interdependent steps, as many as the world's size allows
2. **Logical progression**: each step must follow logically from the previous one
3. **Clear motivation**: each character must have a convincing reason for what they ask
4. **Thematic coherence**: every task must connect to the main story

**MANDATORY TECHNICAL RULES:**
1. **Global connectivity**: the passages you block must NOT leave any location unreachable. Never block the only route to something needed to unblock it
2. **Discoverable solutions**: every puzzle MUST have clues the player can find by exploring, without outside knowledge
3. **Logical obstacles**: the obstacle must be separate from its solution
4. **If a character proposes the puzzle**, the reward must be an item ALREADY IN THAT CHARACTER'S INVENTORY
5. **NO CRAFTING**: the engine does not support creating or transforming items. Do not ask the player to combine or alter objects. Rewards are existing items used directly
6. **OBSTACLE vs. KEY**: `obstacle_name` is what blocks the way (e.g. "locked door"); `required_to_unblock` is the key that removes it (e.g. "iron key"). The key is never the obstacle. `required_to_unblock` may ONLY be of type `item` (an existing item) or `puzzle` (a puzzle you are creating): the engine cannot open a passage any other way
7. **WHO PROPOSES THE PUZZLE**: a puzzle is proposed by a character (`proposed_by_character`) or by an ITEM the player examines. For the latter, put the EXACT name of that ITEM in `proposed_by_location`: despite the field's name, it is NOT a location. The puzzle's location goes in `location`
8. **SHORT ANSWER**: `answer` must be a single word or a very short phrase the player can type as is (e.g. "moon", "1885", "the clock"). The engine looks for that answer in what the player types, so a long explanation makes the puzzle impossible. In `accepted_answers` put 2 to 5 short ways a player could say the SAME answer (the key word alone, without articles, obvious synonyms). E.g. answer "The Silent Echo" → accepted_answers ["echo", "the echo", "silent echo"]
9. **NEW REWARDS**: never reward an item the player already carries at the start, and never block a passage with an item they already carry: they would get past it for free

**MANDATORY HINT RULES:**
For EVERY puzzle you MUST include:
1. **puzzle_hints**: 3-5 progressive hints (general to specific)
2. **interaction_hint**: how to engage with the puzzle if the player has not started
   - Character puzzles: "Try talking to [character]"
   - Item puzzles: "Try examining [item]"
   - Location puzzles: "Look for clues in this location"

**VALIDATION:** before finalizing, check mentally:
1. Is every puzzle solvable with what already exists in the world?
2. Are all locations still reachable after your blocks?
3. Does every element you cite exist, spelled exactly the same?
4. Are the puzzles interesting but not frustrating, and do they fit the theme?
{_language_instruction(language)}"""
