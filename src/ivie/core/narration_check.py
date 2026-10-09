"""Check a turn's narration against the world before the player reads it.

The narrator writes from the engine's facts, yet it still dramatizes what the facts do not say: a
character "hands you the report" on arrival, the player "picks up" a journal they only examined.
A language model asked to judge the narration misses most of these, so the work is split the way
the rest of the turn is: a model extracts what the narration claims (NarrationClaims), and this
module checks each claim against the state before and after the turn. A narration with problems is
rewritten once with them as corrections (core/game_logic.py).
"""

from dataclasses import dataclass, field


@dataclass
class Snapshot:
    """What the player can see and hold, taken before and after a turn."""
    inventory: set = field(default_factory=set)
    lying_here: set = field(default_factory=set)
    held_here: set = field(default_factory=set)
    """items carried by the characters at the player's location"""
    blocked_here: set = field(default_factory=set)
    """destinations of the passages still blocked from the player's location"""


def snapshot(world) -> Snapshot:
    player, location = world.player, world.player.location
    return Snapshot(
        inventory={item.name for item in player.inventory},
        lying_here={item.name for item in location.items},
        held_here={item.name for character in world.characters.values()
                   if character is not player and character.location is location for item in character.inventory},
        blocked_here=set(location.blocked_locations),
    )


def _candidates(name: str, item_names) -> list:
    """The world items a claim may refer to: the one with the same normalized name, or else every
    item whose name contains the claim or is contained in it ("the journal" for "Faded Journal",
    "the key" for both keys). Empty if the claim names nothing in the world."""
    from ..llm.name_matching import normalize_name
    wanted = normalize_name(name or "")
    if not wanted:
        return []
    by_normalized = {normalize_name(item): item for item in item_names}
    if wanted in by_normalized:
        return [by_normalized[wanted]]
    padded = f" {wanted} "
    return [item for normalized, item in by_normalized.items()
            if padded in f" {normalized} " or f" {normalized} " in padded]


def check_claims(claims, before: Snapshot, after: Snapshot, item_names, *, answer_judged: bool,
                 objective_completed: bool, language: str = 'en') -> list:
    """The narration's claims that the state contradicts, written as corrections for the narrator.

    A claim holds if any item it may refer to satisfies it. answer_judged: the engine judged an
    answer in this turn (solved or wrong). objective_completed: in this turn or before.
    """
    es = language == 'es'
    gained = after.inventory - before.inventory
    problems = []
    # A received item the narration describes in its own words ("a crumpled piece of paper" for
    # the Ciphered Note) is one of the items really received that no other claim names
    named = {item for name in claims.items_player_receives + claims.items_characters_offer
             for item in _candidates(name, item_names)}
    unnamed_gains = sorted(gained - named)

    for name in claims.items_player_receives:
        items = _candidates(name, item_names)
        if not items and unnamed_gains:
            unnamed_gains.pop(0)
        elif not items:
            problems.append(f"El jugador no recibe '{name}': no existe en el mundo." if es
                            else f"The player does not receive '{name}': it does not exist in the world.")
        elif not any(item in gained for item in items):
            problems.append((f"El jugador no recibe {items[0]} en este turno: " if es
                             else f"The player does not receive {items[0]} in this turn: ") + _where(items[0], after, es))

    at_hand = before.inventory | after.inventory | before.lying_here | after.lying_here  # given or dropped this turn too
    for name in claims.items_player_holds_or_uses:
        items = _candidates(name, item_names)
        if not any(item in at_hand for item in items):
            shown = items[0] if items else name
            problems.append(f"El jugador no tiene {shown}: no está en su inventario." if es
                            else f"The player does not have {shown}: it is not in their inventory.")

    for name in claims.items_characters_offer:
        items = _candidates(name, item_names)
        if not items and unnamed_gains:
            unnamed_gains.pop(0)
        elif not items:
            problems.append(f"Ningún personaje tiene '{name}': no existe en el mundo." if es
                            else f"No character has '{name}': it does not exist in the world.")
        elif not any(item in before.held_here | after.held_here | gained for item in items):
            problems.append(f"Ningún personaje presente tiene {items[0]}." if es else f"No character here has {items[0]}.")

    for destination in claims.passages_opened:
        still_blocked = _candidates(destination, after.blocked_here)
        if still_blocked:
            problems.append(f"El paso hacia {still_blocked[0]} sigue bloqueado." if es
                            else f"The way to {still_blocked[0]} is still blocked.")

    if claims.answer_judged and not answer_judged:
        problems.append("El jugador no dio ninguna respuesta que el juego evaluara: no digas si es correcta o incorrecta." if es
                        else "The player gave no answer the game judged: do not say whether it is right or wrong.")
    if claims.objective_completed and not objective_completed:
        problems.append("El objetivo no está cumplido: la aventura sigue." if es
                        else "The objective is not completed: the adventure goes on.")
    return problems


def _where(item, after: Snapshot, es: bool) -> str:
    if item in after.inventory:
        return "ya lo tenía." if es else "they already had it."
    if item in after.held_here:
        return "lo sigue teniendo un personaje." if es else "a character still has it."
    if item in after.lying_here:
        return "sigue en el lugar, sin tomar." if es else "it is still lying here, not taken."
    return "no está aquí." if es else "it is not here."


def narration_problems(world, narration: str, before: Snapshot, outcome, objective_completed: bool,
                       language: str, model) -> list:
    """Extract the narration's claims with the model and check them. A failure of the model never
    blocks the turn: the narration is then kept as it is."""
    from ..llm.turn_prompts import prompt_extract_claims
    system_msg, user_msg, schema = prompt_extract_claims(narration, list(world.items), language)
    try:
        response = model.prompt_model_structured(prompt=system_msg + "\n\n" + user_msg, response_schema=schema,
                                                 max_output_tokens=1024)
        claims = schema.model_validate(response) if isinstance(response, dict) else None
    except Exception as error:
        print(f"⚠️ Could not check the narration: {error}")
        return []
    if claims is None:
        return []
    judged = any(kind in ('puzzle_solved', 'puzzle_failed', 'mystery_solved', 'accusation_failed', 'accusation_premature')
                 for kind in outcome.kinds())
    return check_claims(claims, before, snapshot(world), set(world.items), answer_judged=judged,
                        objective_completed=objective_completed, language=language)
