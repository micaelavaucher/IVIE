"""The inspiration screen: Enter or the button creates the world, and an empty idea is refused."""

from streamlit.testing.v1 import AppTest

import src.ivie.ui.streamlit_interface as interface


def _app() -> AppTest:
    app = AppTest.from_file("streamlit_app.py", default_timeout=60)
    app.session_state["language"] = 'es'
    app.session_state["generation_mode"] = 'inspiration'
    return app.run()


def _create_button(app):
    return next(button for button in app.button if button.label == "Crear mundo")


def test_an_idea_creates_the_world(monkeypatch):
    created = []
    monkeypatch.setattr(interface, "generate_world_from_inspiration", created.append)
    app = _app()

    app.text_input(key="inspiration_input").input("  hadas de fuego  ")
    _create_button(app).click().run()

    assert not app.exception and created == ["hadas de fuego"]


def test_the_button_is_always_enabled_and_an_empty_idea_is_refused(monkeypatch):
    created = []
    monkeypatch.setattr(interface, "generate_world_from_inspiration", created.append)
    app = _app()

    assert not _create_button(app).disabled
    _create_button(app).click().run()

    assert created == [] and app.warning[0].value == "✍️ Escribe una idea para tu mundo."


def test_a_suggestion_fills_the_idea():
    app = _app()

    next(button for button in app.button if button.label == "🍄 Super Mario").click().run()

    assert app.text_input(key="inspiration_input").value.startswith("Una aventura colorida en el Reino Champiñón")
