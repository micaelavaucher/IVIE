"""Everything worth seeing about a world, ready to show: the world viewer's data.

world_view(world) reads a world, freshly generated or in play, and describes it for people: the
places and how they connect, who is where, the items, the puzzles with how each one is presented
and its answer, the mystery, and the order in which the world opens up until the objective. It
only reads the world; the viewer (ui/world_viewer.py) decides how to draw it.

The order in which the world opens up comes from the same walk that checks a generated world can
be completed (llm/solvability.py), so the viewer never explains the world with rules of its own.
"""

from dataclasses import asdict, dataclass, field

from .world import MysteryObjective, Puzzle
from ..llm.solvability import progress, unmet_goals
from ..llm.structured_data_models import (
    BlockedPassage, GeneratedCharacter, GeneratedItem, GeneratedLocation, GeneratedPuzzle, GeneratedWorld,
    ItemRequirement, ItemReward, ObjectiveReward, PassageReward, PuzzleRequirement, RequirementType,
)


_TEXTS = {
    'es': {
        'talk': "Hablándole a {character} (en {place})",
        'examine': "Examinando {item} ({where})",
        'cross': "Al intentar pasar de {origin} a {destination}",
        'nobody': "Nada en el mundo lo presenta",
        'at': "en {place}", 'held': "lo tiene {character}", 'carried': "lo llevas tú", 'nowhere': "todavía en ningún lado",
        'passage': "el paso de {origin} a {destination}", 'objective': "completar el objetivo",
        'start': "Empiezas en {place}.", 'start_carrying': "Empiezas en {place} llevando {items}.",
        'reach': "Puedes llegar a {places}.",
        'got_reward': "Consigues {item}, el premio de «{puzzle}».", 'got_given': "{character} te da {item}.",
        'got_given_asks': "{character} te da {item} si cumples lo que pide.", 'got_lying': "Tomas {item} en {place}.",
        'got': "Consigues {item}.",
        'solve': "Resuelves «{puzzle}» ({how}).", 'open_item': "Se abre el paso de {origin} a {destination} con {item}.",
        'open_puzzle': "Se abre el paso de {origin} a {destination} al resolver «{puzzle}».",
        'how_talk': "te lo plantea {character}", 'how_examine': "al examinar {item}", 'how_cross': "al intentar pasar a {destination}",
        'how_unknown': "nada lo presenta",
        'mystery_end': "Examinas {items}, reúnes todas las pistas y acusas al culpable.",
        'done': "El objetivo ya está cumplido.",
        'and': " y ",
    },
    'en': {
        'talk': "Talking to {character} (in {place})",
        'examine': "Examining {item} ({where})",
        'cross': "Trying to go from {origin} to {destination}",
        'nobody': "Nothing in the world presents it",
        'at': "in {place}", 'held': "{character} has it", 'carried': "you carry it", 'nowhere': "nowhere yet",
        'passage': "the way from {origin} to {destination}", 'objective': "completing the objective",
        'start': "You start in {place}.", 'start_carrying': "You start in {place} carrying {items}.",
        'reach': "You can reach {places}.",
        'got_reward': "You get {item}, the prize of «{puzzle}».", 'got_given': "{character} gives you {item}.",
        'got_given_asks': "{character} gives you {item} once you do what they ask.", 'got_lying': "You take {item} in {place}.",
        'got': "You get {item}.",
        'solve': "You solve «{puzzle}» ({how}).", 'open_item': "The way from {origin} to {destination} opens with {item}.",
        'open_puzzle': "The way from {origin} to {destination} opens by solving «{puzzle}».",
        'how_talk': "{character} asks it", 'how_examine': "examining {item}", 'how_cross': "trying to go to {destination}",
        'how_unknown': "nothing presents it",
        'mystery_end': "You examine {items}, gather every clue and accuse the culprit.",
        'done': "The objective is already completed.",
        'and': " and ",
    },
}


def _texts(language: str) -> dict:
    return _TEXTS['es' if language == 'es' else 'en']


def _listing(names, language: str) -> str:
    names = list(names)
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + _texts(language)['and'] + names[-1]


def _description(component) -> str:
    return next((text for text in (getattr(component, 'descriptions', None) or []) if text), "")


#---- What the viewer shows ------------------------------------------------------------

@dataclass
class ItemView:
    name: str
    description: str
    gettable: bool
    where: str
    """a readable place: in a location, with a character, with the player"""
    place: str = None
    """the location it is in, or where its holder is"""


@dataclass
class ExitView:
    destination: str
    blocked_by: str = None
    """the item or puzzle that opens it, if it is closed"""
    blocked_kind: str = None
    """'item' or 'puzzle'"""


@dataclass
class PlaceView:
    name: str
    description: str
    start: bool
    here: bool
    visited: bool
    exits: list
    items: list
    characters: list


@dataclass
class CharacterView:
    name: str
    description: str
    place: str
    carries: list
    gives: str = None
    asks: list = field(default_factory=list)
    proposes: str = None
    says: str = None
    tells: str = None


@dataclass
class PuzzleView:
    name: str
    kind: str
    problem: str
    answer: str
    also_accepted: list
    rewards: list
    presented: list
    """the ways the player gets it presented"""
    hints: list
    state: str
    """'not_proposed', 'proposed' or 'solved'"""


@dataclass
class ClueView:
    name: str
    description: str
    item: str
    where: str
    discovered: bool


@dataclass
class MysteryView:
    question: str
    culprit: str
    solution: str
    clues: list


@dataclass
class EdgeView:
    a: str
    b: str
    blocked_by: str = None
    blocked_kind: str = None


@dataclass
class WorldView:
    language: str
    player: str
    start: str
    here: str
    carried: list
    objective: str
    objective_description: str
    completed: bool
    places: list
    edges: list
    characters: list
    items: list
    puzzles: list
    mystery: MysteryView = None
    path: list = field(default_factory=list)
    """the stages in which the world opens up, each a list of sentences"""
    path_ends: str = ""
    """the last step, the objective itself"""
    path_problems: list = field(default_factory=list)
    """what the objective needs that can never be reached from here"""
    opening: str = ""
    """the opening narration, when there is one"""
    theme: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


#---- Reading the world ----------------------------------------------------------------

def _npcs(world) -> list:
    return [character for character in world.characters.values() if character is not world.player]


def _holder(world, item):
    return next((character for character in world.characters.values() if item in character.inventory), None)


def _lying_at(world, item):
    return next((location for location in world.locations.values() if item in location.items), None)


def _item_where(world, item, language) -> tuple:
    """(readable place, location name) for an item."""
    texts = _texts(language)
    if item in world.player.inventory:
        return texts['carried'], world.player.location.name
    holder = _holder(world, item)
    if holder is not None:
        return texts['held'].format(character=holder.name), holder.location.name
    location = _lying_at(world, item)
    if location is not None:
        return texts['at'].format(place=location.name), location.name
    return texts['nowhere'], None


def _start(world) -> str:
    """Where the player started: the first place they visited."""
    return next(iter(getattr(world.player, 'visited_locations', None) or {}), world.player.location.name)


def _reward_text(reward, language) -> str:
    texts = _texts(language)
    if isinstance(reward, ItemReward):
        return reward.item_name
    if isinstance(reward, PassageReward):
        return texts['passage'].format(origin=reward.from_location, destination=reward.to_location)
    if isinstance(reward, ObjectiveReward):
        return texts['objective']
    return getattr(reward, 'description', None) or str(reward)


def _gated_passages(world, puzzle) -> list:
    """(origin, destination) of the blocked passages this puzzle opens."""
    return [(location.name, destination) for location in world.locations.values()
            for destination, (_, obstacle, _) in location.blocked_locations.items() if obstacle is puzzle]


def puzzle_access(world, puzzle, language: str = 'es') -> list:
    """How the player gets the puzzle presented in this world, one readable way each."""
    texts = _texts(language)
    ways = []
    if puzzle.proposed_by_character:
        character = world.characters.get(puzzle.proposed_by_character)
        place = character.location.name if character else "?"
        ways.append(texts['talk'].format(character=puzzle.proposed_by_character, place=place))
    if puzzle.proposed_by_location:
        item = world._find_object_flexible(puzzle.proposed_by_location)
        where = _item_where(world, item, language)[0] if item is not None and hasattr(item, 'gettable') else texts['nowhere']
        ways.append(texts['examine'].format(item=puzzle.proposed_by_location, where=where))
    for origin, destination in _gated_passages(world, puzzle):
        ways.append(texts['cross'].format(origin=origin, destination=destination))
    return ways or [texts['nobody']]


def _hint_text(hint, language) -> str:
    if isinstance(hint, dict):
        return hint.get('text_es' if language == 'es' else 'text') or hint.get('text') or ""
    return getattr(hint, 'text', None) or ""


def _requirement_text(requirement) -> str:
    if isinstance(requirement, dict):
        return requirement.get('description') or ""
    return getattr(requirement, 'description', None) or ""


def _places(world, here, start) -> list:
    places = []
    for location in world.locations.values():
        exits = [ExitView(destination=other.name) for other in location.connecting_locations]
        for destination, (_, obstacle, _) in location.blocked_locations.items():
            exits.append(ExitView(destination=destination, blocked_by=getattr(obstacle, 'name', str(obstacle)),
                                  blocked_kind='puzzle' if isinstance(obstacle, Puzzle) else 'item'))
        places.append(PlaceView(
            name=location.name, description=_description(location), start=location.name == start,
            here=location.name == here, visited=location.name in (getattr(world.player, 'visited_locations', None) or {}),
            exits=exits, items=[item.name for item in location.items],
            characters=[character.name for character in _npcs(world) if character.location is location]))
    return places


def _edges(world) -> list:
    """Each connection once, closed if it is closed from either side."""
    edges = {}
    for location in world.locations.values():
        for other in location.connecting_locations:
            edges.setdefault(frozenset((location.name, other.name)), EdgeView(a=location.name, b=other.name))
        for destination, (_, obstacle, _) in location.blocked_locations.items():
            edges[frozenset((location.name, destination))] = EdgeView(
                a=location.name, b=destination, blocked_by=getattr(obstacle, 'name', str(obstacle)),
                blocked_kind='puzzle' if isinstance(obstacle, Puzzle) else 'item')
    for puzzle in world.puzzles.values():
        if world.puzzle_states.get(puzzle.name) == 'solved':
            continue
        for reward in puzzle.rewards:
            if isinstance(reward, PassageReward) and frozenset((reward.from_location, reward.to_location)) not in edges:
                if reward.from_location in world.locations and reward.to_location in world.locations:
                    edges[frozenset((reward.from_location, reward.to_location))] = EdgeView(
                        a=reward.from_location, b=reward.to_location, blocked_by=puzzle.name, blocked_kind='puzzle')
    return list(edges.values())


def _characters(world) -> list:
    views = []
    for character in _npcs(world):
        interaction = getattr(character, 'interaction', None)
        view = CharacterView(name=character.name, description=_description(character), place=character.location.name,
                             carries=[item.name for item in character.inventory])
        if interaction is not None and not isinstance(interaction, dict):
            view.gives = getattr(interaction, 'gives_item', None)
            view.asks = [text for text in (_requirement_text(r) for r in (getattr(interaction, 'requires', None) or [])) if text]
            view.proposes = getattr(interaction, 'proposes_puzzle', None)
            view.says = getattr(interaction, 'interaction_text', None)
            view.tells = getattr(interaction, 'gives_information', None)
        views.append(view)
    return views


def _items(world, language) -> list:
    views = []
    for item in world.items.values():
        where, place = _item_where(world, item, language)
        views.append(ItemView(name=item.name, description=_description(item), gettable=item.gettable, where=where, place=place))
    return views


def _puzzles(world, language) -> list:
    return [PuzzleView(
        name=puzzle.name, kind=str(getattr(puzzle.puzzle_type, 'value', puzzle.puzzle_type) or ""),
        problem=puzzle.problem, answer=puzzle.answer, also_accepted=list(getattr(puzzle, 'accepted_answers', None) or []),
        rewards=[_reward_text(reward, language) for reward in puzzle.rewards],
        presented=puzzle_access(world, puzzle, language),
        hints=[text for text in (_hint_text(hint, language) for hint in (puzzle.puzzle_hints or [])) if text],
        state=world.puzzle_states.get(puzzle.name, 'not_proposed'),
    ) for puzzle in world.puzzles.values()]


def _mystery(world, language):
    mystery = world.objective[1] if world.objective and len(world.objective) > 1 else None
    if not isinstance(mystery, MysteryObjective):
        return None
    clues = []
    for clue in mystery.clues:
        item = world._find_item_case_insensitive(clue.associated_item)
        where = _item_where(world, item, language)[0] if item is not None else _texts(language)['nowhere']
        clues.append(ClueView(name=clue.name, description=clue.description, item=clue.associated_item,
                              where=where, discovered=bool(clue.discovered)))
    return MysteryView(question=getattr(mystery, 'question', "") or mystery.description,
                       culprit=getattr(mystery, 'culprit', None), solution=mystery.mystery_solution or "", clues=clues)


#---- How the world opens up -----------------------------------------------------------

def _requirement_for(obstacle):
    if isinstance(obstacle, Puzzle):
        return PuzzleRequirement.model_construct(requirement_type=RequirementType.PUZZLE, description="", puzzle_name=obstacle.name)
    return ItemRequirement.model_construct(requirement_type=RequirementType.ITEM, description="",
                                           item_name=getattr(obstacle, 'name', str(obstacle)))


def _character_structure(character) -> GeneratedCharacter:
    interaction = getattr(character, 'interaction', None)
    return GeneratedCharacter.model_construct(
        name=character.name, descriptions=character.descriptions, location=character.location.name,
        inventory=[item.name for item in character.inventory],
        interaction=interaction if hasattr(interaction, 'requires') else None)


def structure(world) -> GeneratedWorld:
    """The world as it stands now, in the shape the solvability walk reads.

    A world in play keeps no generated description of itself, and its passages and items have moved
    since it was generated, so its current state is described afresh. Only what the walk reads is
    filled in, without validation: it is a view of a world that is already built.
    """
    locations = [GeneratedLocation.model_construct(
        name=location.name, descriptions=location.descriptions, items=[item.name for item in location.items],
        connecting_locations=[other.name for other in location.connecting_locations] + list(location.blocked_locations),
        blocked_passages=[BlockedPassage.model_construct(location=destination, required_to_unblock=_requirement_for(obstacle))
                          for destination, (_, obstacle, _) in location.blocked_locations.items()],
    ) for location in world.locations.values()]
    return GeneratedWorld.model_construct(
        locations=locations,
        items=[GeneratedItem.model_construct(name=item.name, descriptions=item.descriptions, gettable=item.gettable)
               for item in world.items.values()],
        characters=[_character_structure(character) for character in _npcs(world)],
        puzzles=[GeneratedPuzzle.model_construct(
            name=puzzle.name, proposed_by_character=puzzle.proposed_by_character,
            proposed_by_location=puzzle.proposed_by_location,
            rewards=[reward for reward in puzzle.rewards if not isinstance(reward, dict)],
        ) for puzzle in world.puzzles.values()],
        player=_character_structure(world.player),
        objective=getattr(world, 'objective_data', None),
    )


def _how_presented(world, puzzle, language) -> str:
    texts = _texts(language)
    if puzzle.proposed_by_character:
        return texts['how_talk'].format(character=puzzle.proposed_by_character)
    if puzzle.proposed_by_location:
        return texts['how_examine'].format(item=puzzle.proposed_by_location)
    gated = _gated_passages(world, puzzle)
    if gated:
        return texts['how_cross'].format(destination=gated[0][1])
    return texts['how_unknown']


def _reward_of(world, item_name, puzzle_names):
    """The puzzle among these whose prize is the item, if any."""
    return next((name for name in puzzle_names if any(
        isinstance(reward, ItemReward) and reward.item_name == item_name for reward in world.puzzles[name].rewards)), None)


def _obtained(world, name, language) -> tuple:
    """(sentence, given by a character) for an item that is not a prize: someone gives it, or it is
    lying somewhere."""
    texts = _texts(language)
    item = world.items.get(name)
    holder = _holder(world, item) if item is not None else None
    if holder is not None:
        asks = getattr(getattr(holder, 'interaction', None), 'requires', None)
        return texts['got_given_asks' if asks else 'got_given'].format(item=name, character=holder.name), True
    location = _lying_at(world, item) if item is not None else None
    if location is not None:
        return texts['got_lying'].format(item=name, place=location.name), False
    return texts['got'].format(item=name), True


def _opened(world, origin, destination, language) -> tuple:
    """(sentence, opened by a puzzle) for a blocked passage that opens."""
    texts = _texts(language)
    location = world.locations.get(origin)
    obstacle = location.blocked_locations.get(destination, (None, None, None))[1] if location is not None else None
    if obstacle is not None and not isinstance(obstacle, Puzzle):
        return texts['open_item'].format(origin=origin, destination=destination, item=obstacle.name), False
    puzzle = obstacle.name if obstacle is not None else next(
        (p.name for p in world.puzzles.values() if any(isinstance(r, PassageReward) and r.from_location == origin
                                                       and r.to_location == destination for r in p.rewards)), "?")
    return texts['open_puzzle'].format(origin=origin, destination=destination, puzzle=puzzle), True


def _stage_sentences(world, stage, first: bool, language) -> list:
    """One stage of the walk, told in the order the walk finds things (llm/solvability.py): the
    items lying in the places at hand are taken, passages open with them, new places are reached,
    the characters there hand over what they carry, then the puzzles now presented are solved and
    hand out their prizes."""
    texts = _texts(language)
    here = world.player.location.name
    carried = {item.name for item in world.player.inventory}
    sentences = []
    if first:
        sentences.append(texts['start_carrying' if carried else 'start'].format(
            place=here, items=_listing(sorted(carried), language)))
    prizes = {name: _reward_of(world, name, stage.solved) for name in stage.obtained}
    obtained = [_obtained(world, name, language) for name in stage.obtained if not prizes[name] and name not in carried]
    opened = [_opened(world, origin, destination, language) for origin, destination in stage.opened]
    sentences += [sentence for sentence, given in obtained if not given]
    sentences += [sentence for sentence, by_puzzle in opened if not by_puzzle]
    reached = [place for place in stage.reached if place != here]
    if reached:
        sentences.append(texts['reach'].format(places=_listing(reached, language)))
    sentences += [sentence for sentence, given in obtained if given]
    for puzzle_name in stage.solved:
        sentences.append(texts['solve'].format(puzzle=puzzle_name, how=_how_presented(world, world.puzzles[puzzle_name], language)))
        sentences += [texts['got_reward'].format(item=name, puzzle=puzzle_name) for name in stage.obtained if prizes[name] == puzzle_name]
    sentences += [sentence for sentence, by_puzzle in opened if by_puzzle]
    return sentences


def _path(world, language) -> tuple:
    """(stages, last step, problems): the order in which the world opens up until the objective."""
    texts = _texts(language)
    if world.objective and world.check_objective():
        return [], texts['done'], []
    shape = structure(world)
    already = {name for name, state in world.puzzle_states.items() if state == 'solved'}
    state = progress(shape, solved=already)
    stages = [sentences for number, stage in enumerate(state.stages)
              if (sentences := _stage_sentences(world, stage, number == 0, language))]

    mystery = _mystery(world, language)
    if mystery is not None:
        ending = texts['mystery_end'].format(items=_listing([clue.item for clue in mystery.clues], language))
    else:
        ending = world.objective_statement(language) if world.objective else ""
    problems = unmet_goals(shape, state) if shape.objective is not None else []
    return stages, ending, problems


def world_view(world, language: str = 'es', opening: str = "", theme: str = "") -> WorldView:
    """Everything worth seeing about the world, as it stands now."""
    objective_data = getattr(world, 'objective_data', None)
    stages, ending, problems = _path(world, language)
    return WorldView(
        language=language, player=world.player.name, start=_start(world), here=world.player.location.name,
        carried=[item.name for item in world.player.inventory],
        objective=world.objective_statement(language) if world.objective else "",
        objective_description=getattr(objective_data, 'description', "") or "",
        completed=bool(world.objective) and world.check_objective(),
        places=_places(world, world.player.location.name, _start(world)), edges=_edges(world),
        characters=_characters(world), items=_items(world, language), puzzles=_puzzles(world, language),
        mystery=_mystery(world, language), path=stages, path_ends=ending, path_problems=problems,
        opening=opening or "", theme=theme or "",
    )
