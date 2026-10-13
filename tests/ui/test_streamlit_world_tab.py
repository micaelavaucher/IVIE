"""The Streamlit app shows the world viewer next to the game, and in the replay inspector."""

from streamlit.testing.v1 import AppTest

import examples.example_worlds as example_worlds


def _app(**state) -> AppTest:
    app = AppTest.from_file("streamlit_app.py", default_timeout=60)
    for key, value in state.items():
        app.session_state[key] = value
    return app.run()


def test_the_game_has_a_tab_with_the_world_and_the_chat_still_works():
    app = _app(language='es', generation_mode='preset', world=example_worlds.get_world(2, language='es'),
               world_generated=True, chat_history=[{"role": "assistant", "content": "Empieza la aventura."}],
               game_loop=lambda prompt, history: "Miras a tu alrededor.")

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["💬 Partida", "🗺️ El mundo"]
    assert len(app.get("iframe")) == 1

    app.chat_input[0].set_value("mirar").run()

    assert not app.exception and app.chat_message[-1].markdown[0].value == "Miras a tu alrededor."


def test_the_replay_inspector_shows_the_world_viewer():
    app = _app(language='en', generation_mode='replay', inspecting_world=True,
               loaded_world_for_inspection=example_worlds.get_world(1, language='en'))

    assert not app.exception and not app.error
    assert len(app.get("iframe")) == 1
