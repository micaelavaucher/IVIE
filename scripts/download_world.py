"""Download a game from MongoDB to inspect it, using the credentials in .env.

Read-only: it never writes to the database. Run from the repo root:

    .venv/Scripts/python scripts/download_world.py list              # latest games
    .venv/Scripts/python scripts/download_world.py list --limit 30
    .venv/Scripts/python scripts/download_world.py get               # the latest game
    .venv/Scripts/python scripts/download_world.py get WORLD_ID

`get` writes downloads/<world_id>/:
    trace.json   the document as stored (every turn, symbolic states included)
    report.md    the world with its solutions, why each puzzle was or was not presented, and the
                 turns (what the player wrote, the engine's facts, the narration)
    mundo.html   the world viewer for the world as generated: open it in a browser
"""

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUTPUT_DIR = "downloads"


#---- Database ---------------------------------------------------------------------

def _collection():
    from pymongo import MongoClient
    from src.ivie.config import get_database_config, load_config
    uri, db_name, collection_name = get_database_config(load_config())
    return MongoClient(uri, serverSelectionTimeoutMS=15000)[db_name][collection_name]


def _turn_numbers(trace) -> list:
    return sorted(int(key) for key in (trace.get("turns") or {}) if str(key).isdigit())


def list_games(limit):
    fields = {"world_id": 1, "nickname": 1, "created_at": 1, "language": 1, "inspiration": 1, "turns": 1}
    for trace in _collection().find({}, fields).sort("created_at", -1).limit(limit):
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(trace.get("created_at") or 0))
        turns = max(_turn_numbers(trace) or [0])
        theme = trace.get("inspiration") or "(Generar)"
        print(f"{created}  {trace.get('world_id')}  {trace.get('language')}  turnos={turns:<3} "
              f"{trace.get('nickname') or '-'}  {theme}")


#---- Report -----------------------------------------------------------------------

def _decode(encoded):
    import jsonpickle
    return jsonpickle.decode(encoded) if encoded else None


def _name(value):
    return getattr(value, "name", value)


def world_report(world, final_world) -> list:
    from src.ivie.core.world_view import puzzle_access
    lines = [f"Jugador: {world.player.name}, empieza en '{_name(world.player.location)}'"]
    if world.objective:
        lines.append(f"Objetivo (lo que comprueba el motor): {world.objective_statement('es')}")
        mystery = world.objective[1]
        if mystery.__class__.__name__ == "MysteryObjective":
            lines.append(f"Culpable: {getattr(mystery, 'culprit', None)}")
            for clue in mystery.clues:
                lines.append(f"  pista '{clue.name}' en el objeto '{clue.associated_item}'")
    description = getattr(getattr(world, "objective_data", None), "description", None)
    if description:
        lines.append(f"Descripción generada del objetivo: {description}")

    lines.append("\n## Lugares")
    for location in world.locations.values():
        blocked = {destination: _name(obstacle) for destination, (_, obstacle, _) in location.blocked_locations.items()}
        lines.append(f"- **{location.name}**: conecta con {[l.name for l in location.connecting_locations]}; "
                     f"bloqueados (destino: lo que lo abre) {blocked}; objetos {[i.name for i in location.items]}")

    lines.append("\n## Personajes")
    for character in world.characters.values():
        if character is world.player:
            continue
        interaction = getattr(character, "interaction", None)
        details = ""
        if interaction is not None:
            requires = [getattr(r, "description", "") for r in (getattr(interaction, "requires", None) or [])]
            details = (f"; da: {getattr(interaction, 'gives_item', None)}; propone: {getattr(interaction, 'proposes_puzzle', None)}"
                       f"; pide: {requires}; dice: \"{getattr(interaction, 'interaction_text', '') or ''}\"")
        lines.append(f"- **{character.name}** en '{_name(character.location)}', lleva {[i.name for i in character.inventory]}{details}")

    lines.append("\n## Acertijos")
    final_states = getattr(final_world, "puzzle_states", {}) if final_world else {}
    visited = set(getattr(final_world.player, "visited_locations", {}) or {}) if final_world else set()
    for puzzle in world.puzzles.values():
        rewards = [getattr(r, "item_name", None) or f"{getattr(r, 'from_location', '?')} -> {getattr(r, 'to_location', '?')}"
                   for r in puzzle.rewards]
        lines.append(f"- **{puzzle.name}**: \"{puzzle.problem}\"")
        lines.append(f"  - respuesta: {puzzle.answer} (también: {getattr(puzzle, 'accepted_answers', [])})")
        lines.append(f"  - recompensa: {rewards}")
        lines.append(f"  - se presenta: {'; '.join(puzzle_access(world, puzzle, 'es'))}")
        lines.append(f"  - estado al final: {final_states.get(puzzle.name, '?')}")
    if not world.puzzles:
        lines.append("(el mundo no tiene acertijos)")
    lines.append(f"\nLugares visitados en la partida: {sorted(visited) or '?'}")
    return lines


def turns_report(trace) -> list:
    lines = ["\n## Turnos"]
    for number in _turn_numbers(trace):
        if number == 0:
            continue
        turn = trace["turns"][str(number)]
        lines.append(f"\n### Turno {number}\n> {turn.get('user_input', '')}")
        facts = turn.get("engine_facts")
        if facts is not None:
            lines.append("\nHechos del motor:\n" + ("\n".join(f"- {fact}" for fact in facts) or "- (ningún cambio)"))
        for rejected in turn.get("rejected_narrations") or []:
            lines.append(f"\nNarración descartada por el control ({rejected.get('problems')}):\n{rejected.get('narration')}")
        lines.append(f"\nNarración mostrada:\n{turn.get('narration') or turn.get('predicted_outcomes') or ''}")
    return lines


def get_game(world_id):
    collection = _collection()
    trace = (collection.find_one({"world_id": world_id}) if world_id
             else next(collection.find({}).sort("created_at", -1).limit(1), None))
    if trace is None:
        print(f"No encontré ninguna partida{' con world_id ' + world_id if world_id else ''}.")
        return
    world_id = trace["world_id"]
    folder = os.path.join(OUTPUT_DIR, world_id)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "trace.json"), "w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=1, default=str)

    turns = trace.get("turns") or {}
    numbers = _turn_numbers(trace)
    world = _decode((turns.get("0") or {}).get("initial_symbolic_world_state"))
    final_world = _decode((turns.get(str(numbers[-1])) or {}).get("updated_symbolic_world_state")) if numbers and numbers[-1] else world

    lines = [f"# {world_id}", f"Jugador: {trace.get('nickname')} | idioma: {trace.get('language')} | "
             f"tema: {trace.get('inspiration') or '(Generar)'} | turnos: {max(numbers or [0])}\n"]
    lines += world_report(world, final_world) if world is not None else ["(la traza no tiene el mundo inicial)"]
    lines += turns_report(trace)
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    if world is not None:
        from src.ivie.core.world_view import world_view
        from src.ivie.ui.world_viewer import world_viewer_html
        with open(os.path.join(folder, "mundo.html"), "w", encoding="utf-8") as file:
            file.write(world_viewer_html(world_view(world, trace.get("language") or "es")))
    print(f"Partida {world_id} guardada en {folder}/ (report.md, mundo.html y trace.json)")


def main():
    parser = argparse.ArgumentParser(description="Download a game from MongoDB (read-only)")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="latest games")
    listing.add_argument("--limit", type=int, default=10)
    get = commands.add_parser("get", help="download a game (the latest one if no id is given)")
    get.add_argument("world_id", nargs="?")
    args = parser.parse_args()
    if args.command == "list":
        list_games(args.limit)
    else:
        get_game(args.world_id)


if __name__ == "__main__":
    main()
