from src.ivie.llm.name_matching import NameRoster, normalize_name


def test_normalize_handles_acronyms_parentheticals_accents_and_articles():
    assert normalize_name("A.I.D.A. (Automated Integrated Diagnostic Assistant)") == "aida"
    assert normalize_name("A.I.D.A.") == "aida"
    assert normalize_name("La Torre del Reloj") == "torre del reloj"
    assert normalize_name("Jardín") == normalize_name("jardin")
    assert normalize_name("The Brass Key") == "brass key"


def test_a_single_word_that_is_an_article_is_kept():
    assert normalize_name("La") == "la"


def test_canonical_rewrites_to_the_roster_spelling():
    roster = NameRoster(item=["Llave de Hierro"], location=["El Jardín"])

    assert roster.canonical("item", "llave de hierro") == "Llave de Hierro"
    assert roster.canonical("location", "Jardin") == "El Jardín"
    assert len(roster.rewrites) == 2


def test_exact_and_unknown_names_are_left_alone():
    roster = NameRoster(item=["Brass Key"])

    assert roster.canonical("item", "Brass Key") == "Brass Key"
    assert roster.canonical("item", "Silver Key") == "Silver Key"
    assert roster.canonical("item", None) is None
    assert roster.rewrites == []


def test_ambiguous_normalized_names_only_match_exactly():
    roster = NameRoster(item=["La Llave", "Llave"])

    assert roster.canonical("item", "llave") == "llave"
    assert roster.canonical("item", "Llave") == "Llave"


def test_kinds_do_not_mix():
    roster = NameRoster(item=["Bastion"], location=["Bastión"])

    assert roster.canonical("location", "bastion") == "Bastión"
    assert roster.canonical("item", "bastión") == "Bastion"
