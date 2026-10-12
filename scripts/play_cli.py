"""Play IVIE from the command line, one turn per invocation (for automated playtesting).

The game runs through the same code the Streamlit app uses (world generation, world builder,
starting narration and create_game_loop); only the UI is replaced. The world is saved after
every turn, so each command is a separate process and a session can be resumed at any time.

Usage, from the repository root with the project's Python environment:

    python scripts/play_cli.py new --lang es --theme "Piratas"   # Inspiration mode
    python scripts/play_cli.py new --lang en                     # Generate mode (no theme)
    python scripts/play_cli.py turn "examino la nota"            # play one turn
    python scripts/play_cli.py state                             # symbolic state, no LLM call
    python scripts/play_cli.py reveal                            # the whole world (spoilers)
    python scripts/play_cli.py list                              # all sessions

Commands act on the latest session unless --session NAME is given.

Nothing is written to MongoDB. Everything is kept in playtests/<session>/ (git-ignored):
    generated.json   the GeneratedWorld as produced by the pipeline (to rebuild the world)
    world.json       the current World, after the last turn (jsonpickle)
    meta.json        language, theme, mode, turn count, timings
    transcript.md    every input, reply and symbolic state
    engine.log       everything the engine printed (prompts' debug output, [TIMING], warnings)
"""

import argparse
import contextlib
import json
import logging
import os
import re
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # config.ini and logs/ are resolved relative to the working directory

# Playtests stay out of the team's MongoDB, where the real games are kept: blank credentials make
# the game run without a database (load_dotenv does not override variables that are already set)
for _variable in ("MONGO_URI", "MONGO_USER", "MONGO_PASSWORD", "MONGO_CLUSTER"):
    os.environ[_variable] = ""

PLAYTESTS = os.path.join(ROOT, "playtests")
LATEST = os.path.join(PLAYTESTS, "LATEST")


#---- Sessions -----------------------------------------------------------------

def _session_dir(name):
    return os.path.join(PLAYTESTS, name)


def _latest_session():
    try:
        with open(LATEST, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        sys.exit("No session yet. Start one with: python scripts/play_cli.py new --lang es")


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_world(session_dir, world):
    import jsonpickle
    with open(os.path.join(session_dir, "world.json"), "w", encoding="utf-8") as f:
        f.write(jsonpickle.encode(world, unpicklable=True))


def load_world(session_dir):
    import jsonpickle
    with open(os.path.join(session_dir, "world.json"), encoding="utf-8") as f:
        return jsonpickle.decode(f.read())


def _append_transcript(session_dir, text):
    with open(os.path.join(session_dir, "transcript.md"), "a", encoding="utf-8") as f:
        f.write(text + "\n")


@contextlib.contextmanager
def engine_output(session_dir):
    """Send the engine's prints and warnings to engine.log instead of the terminal."""
    with open(os.path.join(session_dir, "engine.log"), "a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            yield


def _quiet_streamlit():
    # The game logic imports streamlit; outside `streamlit run` it warns on every access
    os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
    import streamlit as st
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("streamlit"):
            logging.getLogger(name).setLevel(logging.ERROR)
    st.session_state["nickname"] = "playtest"


def _models():
    from src.ivie.config import load_config, get_model_names
    from src.ivie.llm.models import get_llm
    reasoning_name, narrative_name = get_model_names(load_config())
    return get_llm(reasoning_name), get_llm(narrative_name)


def _rag_enabled():
    from src.ivie.config import load_config, get_enable_rag
    return get_enable_rag(load_config())


#---- Symbolic state -------------------------------------------------------------

def state_report(world, turn, language="es"):
    """The facts of the world, independent of what the narration said."""
    player = world.player
    location = player.location
    npcs = [c for c in world.characters.values() if c is not player]
    here = [c for c in npcs if c.location is location]
    blocked = {name: getattr(obstacle, "name", str(obstacle))
               for name, (_, obstacle, _) in location.blocked_locations.items()}
    puzzles = {name: world.puzzle_states.get(name, "not_proposed") for name in world.puzzles}

    lines = [
        f"--- ESTADO SIMBÓLICO (turno {turn}) ---",
        f"Ubicación: {location.name}",
        f"Conexiones abiertas: {[l.name for l in location.connecting_locations]}",
        f"Pasajes bloqueados desde acá (destino: obstáculo): {blocked}",
        f"Inventario: {[i.name for i in player.inventory]}",
        f"Objetos acá: {[i.name for i in location.items]}",
        f"Personajes acá: {[f'{c.name} (lleva: {[i.name for i in c.inventory]})' for c in here]}",
        f"Puzzles: {puzzles}",
    ]
    objective_data = getattr(world, "objective_data", None)
    description = getattr(objective_data, "description", None) or "(sin descripción)"
    if world.objective:
        lines.append(f"Objetivo: {world.objective_statement(language)}")
        lines.append(f"Descripción generada del objetivo: {description}")
        target = world.objective[1]
        if target.__class__.__name__ == "MysteryObjective":
            found, total = target.get_completion_progress()
            lines.append(f"Pistas del misterio descubiertas: {found}/{total}")
        lines.append(f"Objetivo cumplido: {world.check_objective()}")
    else:
        lines.append("Objetivo: NINGUNO (el mundo no tiene objetivo construido)")
    return "\n".join(lines)


def reveal_report(world, generated):
    """The whole world, including solutions."""
    player = world.player
    lines = ["=== MUNDO COMPLETO (spoilers) ===", f"Jugador: {player.name}, empieza en '{generated['player']['location']}'"]
    for location in world.locations.values():
        blocked = {name: getattr(obstacle, "name", str(obstacle))
                   for name, (_, obstacle, _) in location.blocked_locations.items()}
        lines.append(f"\nUBICACIÓN {location.name}")
        lines.append(f"  conexiones abiertas: {[l.name for l in location.connecting_locations]}")
        lines.append(f"  bloqueos (destino: lo que lo abre): {blocked}")
        lines.append(f"  objetos: {[i.name for i in location.items]}")
    for character in world.characters.values():
        if character is player:
            continue
        interaction = character.interaction
        lines.append(f"\nPERSONAJE {character.name} en '{character.location.name}', lleva {[i.name for i in character.inventory]}")
        if interaction is not None:
            lines.append(f"  da: {getattr(interaction, 'gives_item', None)} | propone: {getattr(interaction, 'proposes_puzzle', None)}")
            requires = [r.model_dump(exclude_none=True) for r in getattr(interaction, 'requires', [])]
            if requires:
                lines.append(f"  pide: {requires}")
    for puzzle in world.puzzles.values():
        rewards = [r.model_dump(exclude_none=True) if hasattr(r, "model_dump") else r for r in puzzle.rewards]
        lines.append(f"\nPUZZLE {puzzle.name} ({puzzle.puzzle_type}) - propuesto por personaje: {puzzle.proposed_by_character} "
                     f"| por objeto: {puzzle.proposed_by_location}")
        lines.append(f"  problema: {puzzle.problem}")
        lines.append(f"  respuesta: {puzzle.answer}")
        lines.append(f"  recompensas: {rewards}")
    objective = generated["objective"]
    lines.append(f"\nOBJETIVO ({objective['type']}): {objective['description']}")
    lines.append(f"  componentes: {[(c['name'], c['component_type']) for c in objective['components']]}")
    for clue in objective.get("mystery_clues") or []:
        lines.append(f"  pista '{clue['name']}': objeto '{clue['associated_item']}' en '{clue['item_location']}'")
    if objective.get("mystery_question"):
        lines.append(f"  pregunta del misterio: {objective['mystery_question']}")
    if objective.get("mystery_culprit"):
        lines.append(f"  culpable: {objective['mystery_culprit']}")
    if objective.get("mystery_solution"):
        lines.append(f"  solución del misterio: {objective['mystery_solution']}")
    return "\n".join(lines)


#---- Commands -------------------------------------------------------------------

def new_session(language, theme):
    _quiet_streamlit()
    slug = re.sub(r"[^a-z0-9]+", "-", (theme or "generate").lower()).strip("-")[:30]
    name = time.strftime("%Y%m%d-%H%M%S") + f"-{language}-{slug}"
    session_dir = _session_dir(name)
    os.makedirs(session_dir)
    with open(LATEST, "w", encoding="utf-8") as f:
        f.write(name)

    meta = {"session": name, "language": language, "theme": theme,
            "mode": "inspiration" if theme else "generate", "world_id": f"playtest_{name}",
            "turns": 0, "completed": False}
    _write_json(os.path.join(session_dir, "meta.json"), meta)

    started = time.perf_counter()
    try:
        with engine_output(session_dir):
            from src.ivie.llm.generation_pipeline import create_world_incrementally, create_world_incrementally_generate
            from src.ivie.core.world_builder import create_world_from_llm_response
            from src.ivie.core.game_logic import generate_starting_narration
            reasoning, narrative = _models()
            if theme:
                generated = create_world_incrementally(theme, language)
            else:
                generated = create_world_incrementally_generate(language)
            _write_json(os.path.join(session_dir, "generated.json"), generated.model_dump(mode="json"))
            world = create_world_from_llm_response(generated)
            narration = generate_starting_narration(world, language, narrative)
    except Exception:
        meta["error"] = traceback.format_exc()
        _write_json(os.path.join(session_dir, "meta.json"), meta)
        print(f"ERROR: la generación del mundo falló (sesión {name}). Detalle en playtests/{name}/engine.log y meta.json")
        print(traceback.format_exc().strip().splitlines()[-1])
        sys.exit(1)

    meta["generation_seconds"] = round(time.perf_counter() - started, 1)
    _write_json(os.path.join(session_dir, "meta.json"), meta)
    save_world(session_dir, world)

    state = state_report(world, 0, language)
    _append_transcript(session_dir, f"# {name}\n\nTema: {theme or '(modo Generate)'} | idioma: {language} | "
                                    f"generación: {meta['generation_seconds']}s\n\n## Turno 0\n\n{narration}\n\n```\n{state}\n```\n")
    print(f"SESIÓN: {name}  (mundo generado en {meta['generation_seconds']}s)\n")
    print(narration)
    print("\n" + state)


def play_turn(session_name, text, reasoning=None, narrative=None, enable_rag=None):
    """Run one turn of the session. Models and RAG come from config.ini unless given."""
    _quiet_streamlit()
    session_dir = _session_dir(session_name)
    meta = _read_json(os.path.join(session_dir, "meta.json"))
    if "error" in meta:
        sys.exit("Esta sesión no tiene mundo (la generación falló). Empezá otra con 'new'.")
    world = load_world(session_dir)
    turn = meta["turns"] + 1

    started = time.perf_counter()
    engine_error = None
    with engine_output(session_dir):
        print(f"\n===== TURNO {turn}: {text}")
        from src.ivie.core.game_logic import create_game_loop
        if reasoning is None or narrative is None:
            reasoning, narrative = _models()
        rag = _rag_enabled() if enable_rag is None else enable_rag
        try:
            loop = create_game_loop(world, reasoning, narrative, meta["language"], set(), enable_rag=rag,
                                    inspiration=meta.get("theme") or "", world_id=meta["world_id"], start_turn=meta["turns"])
            reply = loop(text, [])
        except Exception:
            engine_error = traceback.format_exc()
            print(engine_error)
            reply = ""
    elapsed = time.perf_counter() - started

    meta["turns"] = turn
    completed_before = meta["completed"]
    if world.objective and world.check_objective():
        meta["completed"] = True
    _write_json(os.path.join(session_dir, "meta.json"), meta)
    save_world(session_dir, world)

    reply = reply.replace(r"\<", "<").replace(r"\>", ">")
    state = state_report(world, turn, meta["language"])
    error_note = f"\n\nERROR DEL MOTOR:\n{engine_error}" if engine_error else ""
    _append_transcript(session_dir, f"## Turno {turn} ({elapsed:.1f}s)\n\n> {text}\n\n{reply}{error_note}\n\n```\n{state}\n```\n")

    print(f"[turno {turn} - {elapsed:.1f}s]\n")
    if engine_error:
        print("ERROR DEL MOTOR (excepción no capturada durante el turno):")
        print(engine_error.strip().splitlines()[-1])
    print(reply)
    print("\n" + state)
    if meta["completed"] and not completed_before:
        print("\n🏁 OBJETIVO COMPLETADO")


def show_state(session_name):
    _quiet_streamlit()
    session_dir = _session_dir(session_name)
    meta = _read_json(os.path.join(session_dir, "meta.json"))
    print(f"SESIÓN: {session_name}")
    print(state_report(load_world(session_dir), meta["turns"], meta["language"]))


def reveal(session_name):
    _quiet_streamlit()
    session_dir = _session_dir(session_name)
    world = load_world(session_dir)
    generated = _read_json(os.path.join(session_dir, "generated.json"))
    _append_transcript(session_dir, "_(el jugador usó `reveal`)_\n")
    print(reveal_report(world, generated))


def list_sessions():
    if not os.path.isdir(PLAYTESTS):
        print("No hay sesiones.")
        return
    for name in sorted(os.listdir(PLAYTESTS)):
        meta_path = os.path.join(PLAYTESTS, name, "meta.json")
        if os.path.isfile(meta_path):
            meta = _read_json(meta_path)
            status = "ERROR" if "error" in meta else ("COMPLETADO" if meta.get("completed") else "en curso")
            print(f"{name}  turnos={meta['turns']}  {status}")


def main():
    parser = argparse.ArgumentParser(description="Play IVIE from the command line, one turn per call.")
    parser.add_argument("--session", help="session name (default: the latest one)")
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="generate a new world")
    new.add_argument("--lang", choices=["es", "en"], default="es")
    new.add_argument("--theme", default="", help="inspiration theme; omit for Generate mode")
    turn = commands.add_parser("turn", help="play one turn")
    turn.add_argument("text")
    commands.add_parser("state", help="show the symbolic state")
    commands.add_parser("reveal", help="show the whole world, including solutions")
    commands.add_parser("list", help="list sessions")
    args = parser.parse_args()

    # Output piped on Windows defaults to cp1252, which cannot print the narration's accents and emojis
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.makedirs(PLAYTESTS, exist_ok=True)
    if args.command == "new":
        new_session(args.lang, args.theme.strip())
    elif args.command == "list":
        list_sessions()
    else:
        session = args.session or _latest_session()
        if args.command == "turn":
            play_turn(session, args.text)
        elif args.command == "state":
            show_state(session)
        elif args.command == "reveal":
            reveal(session)


if __name__ == "__main__":
    main()
