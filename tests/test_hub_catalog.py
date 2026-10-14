"""The bundled hub catalog: entries, cards, search, and serialisation."""

import json

from spikeforge_hub.catalog import catalog as entry_cards
from spikeforge_hub.catalog import (
    entries,
    get,
    issues,
    list_entries,
    search,
)
from spikeforge_hub.entry import FRAMEWORKS, UNVERIFIED_CANDIDATE


def test_catalog_has_at_least_ten_entries_across_frameworks() -> None:
    """The bundled catalog meets the honest baseline: verified entries only."""
    found = entries()
    frameworks = {entry.framework for entry in found}
    assert len(found) >= 10
    assert len(frameworks) >= 5
    assert frameworks <= set(FRAMEWORKS)


def test_bundled_catalog_has_no_validation_issues() -> None:
    """Every bundled entry passes schema validation."""
    assert issues() == []


def test_bundled_catalog_ships_only_verified_licenses() -> None:
    """No shipped entry presents an unverified or placeholder license."""
    assert entries()
    for entry in entries():
        assert entry.license != UNVERIFIED_CANDIDATE
        assert entry.license != "see upstream"
        assert " " not in entry.license


def test_entry_cards_carry_availability_and_cache_state() -> None:
    """Each card exposes the additive availability, reason, and cache keys."""
    for card in entry_cards():
        assert isinstance(card["available"], bool)
        assert isinstance(card["cached"], bool)
        assert card["reason"] is None or isinstance(card["reason"], str)
        assert {"id", "framework", "kind"} <= set(card)


def test_catalog_is_json_serialisable() -> None:
    """The catalog survives ``json.dumps`` for the WebSocket payload."""
    assert isinstance(json.dumps(entry_cards()), str)


def test_list_entries_filters_by_framework() -> None:
    """A framework filter returns only that framework's cards."""
    cards = list_entries(framework="nir")
    assert cards
    assert all(card["framework"] == "nir" for card in cards)


def test_get_returns_entry_or_none() -> None:
    """``get`` resolves a known id and reports an unknown one as ``None``."""
    first = entries()[0]
    assert get(first.id) is first
    assert get("does/not-exist") is None


def test_search_matches_id_and_rejects_empty() -> None:
    """Search matches catalog text and an empty query yields nothing."""
    assert search("conv")
    assert search("   ") == []
