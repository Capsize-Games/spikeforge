"""Dataset provenance: full registry coverage, and no inferred licences.

A trained entry's ``license`` describes the weights; the training data's own
terms are a separate fact with a separate holder. These tests hold the two
apart and hold the table honest: every registry dataset is described, a licence
is either a concrete SPDX-style id or the explicit unverified marker, and every
dataset carries a credit line even where its licence could not be confirmed.
"""

import pytest

from spikeforge.data import datasets
from spikeforge.data.dataset_provenance import (
    UNVERIFIED,
    dataset_provenance,
    provenance_names,
)
from spikeforge_hub.entry import UNVERIFIED_CANDIDATE, HubEntry
from spikeforge_hub.errors import HubCatalogError

#: Licences read from the publisher's own page while writing this table.
_VERIFIED = {
    "fashion": "MIT",
    "kmnist": "CC-BY-SA-4.0",
    "n_mnist": "CC-BY-SA-4.0",
    "dvs128_gesture": "CC-BY-4.0",
    "ssc": "CC-BY-4.0",
}


def test_every_registry_dataset_has_provenance() -> None:
    """The table cannot silently fall behind the registry beside it."""
    missing = [
        name
        for name in datasets.dataset_names()
        if name not in provenance_names()
    ]
    assert missing == []


def test_the_unverified_marker_matches_the_hub_spelling() -> None:
    """Core cannot import the hub, so the duplicated string is pinned here."""
    assert UNVERIFIED == UNVERIFIED_CANDIDATE


@pytest.mark.parametrize(("name", "license_id"), sorted(_VERIFIED.items()))
def test_verified_licences_are_recorded_concretely(
    name: str, license_id: str
) -> None:
    """A licence read from the publisher is stored as a concrete SPDX id."""
    assert dataset_provenance(name).license == license_id


def test_unconfirmed_licences_carry_the_marker_not_a_guess() -> None:
    """MNIST and CIFAR10-DVS could not be confirmed, and say so."""
    for name in ("mnist", "cifar10_dvs"):
        assert dataset_provenance(name).license == UNVERIFIED


def test_every_dataset_carries_a_credit_and_an_audit_trail() -> None:
    """Origin is knowable even where the licence is not, so it is recorded."""
    for name, provenance in provenance_names().items():
        assert provenance.attribution.strip(), name
        assert provenance.source.strip(), name


def test_no_licence_is_free_text() -> None:
    """Every recorded licence passes the same rule the catalog enforces."""
    for name, provenance in provenance_names().items():
        data = {
            "id": f"nir/{name}",
            "name": name,
            "framework": "nir",
            "kind": "nir_graph",
            "source": "bundled",
            "license": "BSD-3-Clause",
            "notes": "probe",
            "dataset_license": provenance.license,
        }
        # Raises if the recorded string is neither a concrete id nor the
        # marker -- "CC-BY-SA-3.0, probably" would fail here.
        HubEntry.from_dict(data)


def test_kmnist_carries_the_wording_codh_asks_for() -> None:
    """CODH requests specific wording; it is reproduced, not paraphrased."""
    attribution = dataset_provenance("kmnist").attribution
    assert "KMNIST Dataset" in attribution
    assert "CODH" in attribution
    assert "Kuzushiji Dataset" in attribution
    assert "doi:10.20676/00000341" in attribution


def test_an_unknown_dataset_raises_rather_than_defaulting() -> None:
    """A dataset with no recorded provenance is not publishable."""
    with pytest.raises(KeyError):
        dataset_provenance("not_a_dataset")


def test_sequence_toy_is_this_projects_own_data() -> None:
    """The synthetic dataset needs no third-party credit, and says so."""
    provenance = dataset_provenance("sequence_toy")
    assert provenance.license == "BSD-3-Clause"
    assert "this project" in provenance.attribution


def test_free_text_dataset_licence_is_rejected_by_name() -> None:
    """The escape hatch closed for `license` is closed here too."""
    data = {
        "id": "nir/probe",
        "name": "probe",
        "framework": "nir",
        "kind": "nir_graph",
        "source": "bundled",
        "license": "BSD-3-Clause",
        "notes": "probe",
        "dataset_license": "see upstream",
    }
    with pytest.raises(HubCatalogError) as error:
        HubEntry.from_dict(data)
    assert "dataset_license" in str(error.value)
