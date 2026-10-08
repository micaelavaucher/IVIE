"""Test configuration.

Tests never touch external services: the MongoDB variables are blanked before any project module
is imported (mongodb_handler connects at import time, and load_dotenv does not override variables
that are already set), so game loops created in tests do not write traces to a real database.
"""

import os

for _name in ("MONGO_URI", "MONGO_USER", "MONGO_PASSWORD", "MONGO_CLUSTER"):
    os.environ[_name] = ""
