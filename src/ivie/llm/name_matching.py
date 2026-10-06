"""Tolerant matching of LLM-written names against a fixed roster of world names.

The skeleton fixes every name up front, but the model does not always copy them exactly:
"A.I.D.A. (Asistente de Diagnóstico)" comes back as "A.I.D.A.", "La Torre del Reloj" as
"Torre del reloj", "Jardín" as "Jardin". Exact matching turns each of those into a failed
verification or, in the world builder, a reference that is silently dropped.

Names are compared on a normalized form and rewritten to the roster's exact spelling. A name
that matches nothing is left untouched, so the verifiers still report it.
"""

import re
import unicodedata
from typing import Dict, Iterable, Optional

_LEADING_ARTICLES = {"the", "a", "an", "el", "la", "los", "las", "lo", "un", "una", "unos", "unas"}


def normalize_name(name: str) -> str:
    """Lowercase, no accents, no parentheticals, no punctuation, no leading article.

    Dotted acronyms collapse ("A.I.D.A." -> "aida"). Returns "" for an empty name.
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace(".", "")
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    if len(tokens) > 1 and tokens[0] in _LEADING_ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


class NameRoster:
    """The exact names of each kind of world element, with lookup by normalized form."""

    def __init__(self, **names_by_kind: Iterable[str]) -> None:
        self._exact: Dict[str, set] = {}
        self._normalized: Dict[str, Dict[str, Optional[str]]] = {}
        self.rewrites = []
        """(kind, written, canonical) for every name that was rewritten"""
        for kind, names in names_by_kind.items():
            self.add(kind, names)

    def add(self, kind: str, names: Iterable[str]) -> None:
        exact = self._exact.setdefault(kind, set())
        normalized = self._normalized.setdefault(kind, {})
        for name in names:
            exact.add(name)
            key = normalize_name(name)
            if not key:
                continue
            # Two roster names that normalize alike are ambiguous: only exact matches count for them
            normalized[key] = None if key in normalized and normalized[key] != name else name

    def names(self, kind: str) -> set:
        return set(self._exact.get(kind, set()))

    def canonical(self, kind: str, name: Optional[str]) -> Optional[str]:
        """The roster spelling of `name`, or `name` unchanged if it matches nothing."""
        if not name or name in self._exact.get(kind, set()):
            return name
        match = self._normalized.get(kind, {}).get(normalize_name(name))
        if match is None:
            return name
        self.rewrites.append((kind, name, match))
        return match

    def canonical_list(self, kind: str, names: Iterable[str]) -> list:
        return [self.canonical(kind, name) for name in names]

    def log_rewrites(self, prefix: str = "[NAMES]") -> None:
        for kind, written, canonical in self.rewrites:
            print(f"{prefix} {kind} '{written}' -> '{canonical}'")
        self.rewrites = []
