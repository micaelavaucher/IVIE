"""world_view: what the world viewer shows about a world, freshly generated or in play."""

import json

from src.ivie.core.world import Character, Item, Location, MysteryClue, MysteryObjective, Puzzle, World
from src.ivie.core.world_view import puzzle_access, world_view
from src.ivie.llm.structured_data_models import CharacterInteraction, GeneratedObjective, ItemReward
from src.ivie.ui.world_viewer import world_viewer_html


def _world(*, key_behind_its_door=False):
    """Hall (start) - Torre; Hall -> Despacho closed by the Llave, which Marlowe gives. The Sabio, in
    the Torre, asks a riddle whose prize is the Gema. The objective: get the Diario, in the Despacho."""
    hall, torre, despacho = Location("Hall", ["Un hall"]), Location("Torre", ["Una torre"]), Location("Despacho", ["d"])
    for a, b in ((hall, torre), (hall, despacho)):
        a.connecting_locations.append(b)
        b.connecting_locations.append(a)
    llave, diario, gema = Item("Llave", ["Una llave"]), Item("Diario", ["d"]), Item("Gema", ["d"])
    estatua = Item("Estatua", ["d"], gettable=False)
    hall.items.append(estatua)
    despacho.items.append(diario)
    hall.block_passage(despacho, llave)
    player = Character("Jugador", ["d"], location=hall)
    marlowe = Character("Marlowe", ["Un mayordomo"], location=despacho if key_behind_its_door else hall, inventory=[llave],
                        interaction=CharacterInteraction(gives_item="Llave", interaction_text="Tome la llave.",
                                                         relevance_to_objective="r"))
    sabio = Character("Sabio", ["d"], location=torre, inventory=[gema],
                      interaction=CharacterInteraction(proposes_puzzle="Acertijo", relevance_to_objective="r"))
    world = World(player)
    world.add_locations([hall, torre, despacho])
    world.add_items([llave, diario, gema, estatua])
    world.add_characters([marlowe, sabio])
    world.add_puzzle(Puzzle("Acertijo", ["d"], problem="¿Qué tiene agujas y no pincha?", answer="reloj",
                            proposed_by_character="Sabio", accepted_answers=["el reloj"],
                            rewards=[ItemReward(description="d", item_name="Gema")]))
    world.objective = (player, diario)
    world.objective_data = GeneratedObjective.model_validate({
        "type": "get_item", "description": "Encuentra el diario", "success_conditions": ["s"],
        "components": [{"name": "Diario", "component_type": "item", "role_in_objective": "g"}]})
    return world


def _place(view, name):
    return next(place for place in view.places if place.name == name)


def test_the_places_say_where_everyone_is_and_what_closes_each_way():
    view = world_view(_world(), 'es')

    hall = _place(view, "Hall")
    assert hall.start and hall.here and hall.characters == ["Marlowe"] and hall.items == ["Estatua"]
    closed = next(exit for exit in hall.exits if exit.destination == "Despacho")
    assert (closed.blocked_by, closed.blocked_kind) == ("Llave", "item")
    assert {(e.a, e.b, e.blocked_by) for e in view.edges} == {("Hall", "Torre", None), ("Hall", "Despacho", "Llave")}
    assert next(item for item in view.items if item.name == "Llave").where == "lo tiene Marlowe"


def test_a_puzzle_shows_how_it_is_presented_its_answer_and_its_prize():
    view = world_view(_world(), 'es')

    puzzle = view.puzzles[0]
    assert puzzle.presented == ["Hablándole a Sabio (en Torre)"]
    assert (puzzle.answer, puzzle.also_accepted, puzzle.rewards) == ("reloj", ["el reloj"], ["Gema"])
    assert puzzle.state == 'not_proposed'


def test_a_puzzle_that_closes_a_way_is_presented_by_trying_to_cross_it():
    world = _world()
    puzzle = world.puzzles["Acertijo"]
    puzzle.proposed_by_character = None
    world.locations["Hall"].block_passage(world.locations["Torre"], puzzle)

    assert puzzle_access(world, puzzle, 'es') == ["Al intentar pasar de Hall a Torre"]


def test_the_path_tells_the_order_in_which_the_world_opens_up():
    view = world_view(_world(), 'es')

    sentences = [sentence for stage in view.path for sentence in stage]
    gets_key, opens = sentences.index("Marlowe te da Llave."), sentences.index("Se abre el paso de Hall a Despacho con Llave.")
    assert sentences[0] == "Empiezas en Hall." and gets_key < opens
    assert "Resuelves «Acertijo» (te lo plantea Sabio)." in sentences
    assert "Consigues Gema, el premio de «Acertijo»." in sentences
    assert view.path_ends == "Tienes que conseguir Diario." and view.path_problems == []


def test_a_world_in_play_is_shown_as_it_stands_now():
    world = _world()
    world.player.move(world.locations["Torre"])
    world.puzzle_states["Acertijo"] = 'solved'
    world.player.save_item(world.items["Gema"], world.characters["Sabio"])

    view = world_view(world, 'es')

    assert (view.start, view.here, view.carried) == ("Hall", "Torre", ["Gema"])
    assert _place(view, "Torre").here and not _place(view, "Hall").here
    sentences = [sentence for stage in view.path for sentence in stage]
    assert sentences[0] == "Empiezas en Torre llevando Gema."
    assert not any("Acertijo" in sentence for sentence in sentences)


def test_a_completed_objective_needs_no_path():
    world = _world()
    world.player.inventory.append(world.items["Diario"])

    view = world_view(world, 'es')

    assert view.completed and view.path == [] and view.path_ends == "El objetivo ya está cumplido."


def test_an_objective_out_of_reach_is_reported():
    view = world_view(_world(key_behind_its_door=True), 'es')

    assert view.path_problems == ["the item 'Diario' can never be obtained"]


def test_a_mystery_shows_its_question_clues_and_culprit():
    world = _world()
    clue = MysteryClue("Tinta", "El diario tiene tinta fresca", "Diario", "r")
    world.objective = (world.player, MysteryObjective("Misterio", "d", [clue], "Marlowe lo hizo", culprit="Marlowe",
                                                      question="¿Quién robó la gema?"))

    view = world_view(world, 'es')

    assert (view.mystery.question, view.mystery.culprit) == ("¿Quién robó la gema?", "Marlowe")
    assert view.mystery.clues[0].where == "en Despacho" and not view.mystery.clues[0].discovered
    assert view.path_ends == "Examinas Diario, reúnes todas las pistas y acusas al culpable."


def test_the_viewer_page_carries_the_world_safely():
    world = _world()
    world.locations["Torre"].descriptions = ["</script><script>alert(1)</script>"]

    html = world_viewer_html(world_view(world, 'es', opening="**Despiertas** en el hall."))

    assert "/*__WORLD__*/" not in html and "</script><script>alert" not in html
    data = html.split("const WORLD = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(data)["places"][1]["description"] == "</script><script>alert(1)</script>"
