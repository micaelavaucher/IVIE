"""The world viewer: a self-contained HTML page that shows a world.

core/world_view.py describes the world; this module puts that description into the page template
(static/world_viewer.html), which draws it. The page needs nothing else, so it can be shown inside
Streamlit or saved as a file and opened in a browser.
"""

import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'static', 'world_viewer.html')
_PLACEHOLDER = "/*__WORLD__*/null"


def _script_json(data) -> str:
    """JSON that is safe inside a <script>: no '</script>' or line separators can end it early."""
    text = json.dumps(data, ensure_ascii=False)
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def world_viewer_html(view) -> str:
    """The viewer page for a WorldView (core/world_view.py)."""
    with open(TEMPLATE_PATH, encoding='utf-8') as file:
        template = file.read()
    return template.replace(_PLACEHOLDER, _script_json(view.to_dict()), 1)
