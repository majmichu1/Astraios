"""Tests for the per-target recipe book."""

import json

from cosmica.core.recipe import RecipeBook, get_recipe_book


def test_type_recipe_applies():
    rb = get_recipe_book()
    # A reflection nebula with no catalog hints gets the type recipe.
    merged = rb.resolve("reflection_nebula", "SomeRefNeb", {})
    assert merged.get("reflection_nebulosity") is True
    assert merged.get("stretch") == "gentle"
    assert merged.get("chroma_strength", 0) >= 1.0


def test_catalog_hints_override_type():
    rb = get_recipe_book()
    # Catalog hint beats the type default.
    merged = rb.resolve("reflection_nebula", "X", {"stretch": "aggressive"})
    assert merged.get("stretch") == "aggressive"
    # …but unspecified type keys still come through.
    assert merged.get("reflection_nebulosity") is True


def test_named_override_wins():
    rb = get_recipe_book()
    merged = rb.resolve("reflection_nebula", "NGC 7023", {"stretch": "moderate"})
    # The named NGC 7023 override sets these.
    assert merged.get("use_starnet") is True
    assert merged.get("chroma_strength") == 2.0


def test_named_lookup_is_space_insensitive():
    rb = get_recipe_book()
    assert rb.has_named("ngc7023")
    assert rb.has_named("NGC 7023")
    assert rb.has_named("NGC7023")
    assert not rb.has_named("M101")


def test_unknown_type_falls_through():
    rb = get_recipe_book()
    merged = rb.resolve("totally_unknown_type", "Z", {"stretch": "gentle"})
    assert merged == {"stretch": "gentle"}  # only the catalog hint


def test_missing_recipes_file_is_safe(tmp_path):
    rb = RecipeBook(path=tmp_path / "nope.json")
    merged = rb.resolve("galaxy_spiral", "M51", {"foo": 1})
    assert merged == {"foo": 1}  # heuristics-only, no crash


def test_every_catalog_object_type_has_a_recipe():
    # The type recipes should cover the object types the catalog actually uses,
    # so no real target falls through to bare heuristics.
    from cosmica.core.catalog import _CATALOG_JSON

    with open(_CATALOG_JSON, encoding="utf-8") as fh:
        catalog = json.load(fh)
    used_types = {e.get("object_type") for e in catalog}
    rb = get_recipe_book()
    rb._ensure_loaded()
    missing = used_types - set(rb._types)
    assert not missing, f"object types with no type-recipe: {missing}"
