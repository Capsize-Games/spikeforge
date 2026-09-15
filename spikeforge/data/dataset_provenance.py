"""What each registry dataset's own terms permit, and who to credit.

A trained checkpoint's ``license`` describes **the weights** — this project's
own artifact, BSD-3-Clause. It says nothing about the data those weights
encode, and the two are not the same question: KMNIST is CC BY-SA 4.0 with a
specific requested wording, while a checkpoint trained on it is this project's
to license.

Whether trained weights are "adapted material" under a ShareAlike licence is
genuinely unsettled — the prevailing ML norm says they are not, and Creative
Commons state their licences are not designed to govern model weights. This
module takes no position on that. It removes the need to have one: record what
the upstream terms are, carry the attribution the publisher asks for, and let
a reader judge.

This table lives beside the registry it describes so the two cannot drift;
``tests/test_dataset_provenance.py`` fails if a registry dataset has no entry
here.

**Verification discipline.** ``license`` is a concrete SPDX-style id only when
it was read from the publisher's own page. A dataset whose terms could not be
confirmed from a primary source carries
:data:`~spikeforge_hub.entry.UNVERIFIED_CANDIDATE` instead — never a licence
inferred from secondary sources, and never free text. ``attribution`` is a
statement of origin rather than a legal term, so it is recorded even where the
licence is not: who made the data is knowable regardless.
"""

from dataclasses import dataclass
from typing import Dict

#: Marker for a dataset whose upstream terms are not confirmed from a primary
#: source. Deliberately a duplicate of
#: ``spikeforge_hub.entry.UNVERIFIED_CANDIDATE`` rather than an import:
#: ``spikeforge-hub`` is only an optional dependency of core (the ``all``
#: extra), so core cannot import it. ``tests/test_dataset_provenance.py``
#: asserts the two strings stay equal.
UNVERIFIED = "unverified-candidate"


@dataclass(frozen=True)
class DatasetProvenance:
    """One dataset's upstream licence and the credit its publisher asks for."""

    #: A concrete SPDX-style id, or :data:`UNVERIFIED` when the publisher's
    #: own terms could not be read. Never free text, never a guess.
    license: str
    #: The credit line to carry. Where the publisher requests specific
    #: wording, this is that wording verbatim.
    attribution: str
    #: Why the licence field says what it says: the page it was read from, or
    #: what blocked confirmation. This is the audit trail for the claim.
    source: str


_PROVENANCE: Dict[str, DatasetProvenance] = {
    # --- verified against the publisher's own page ----------------------
    "fashion": DatasetProvenance(
        license="MIT",
        attribution=(
            "Fashion-MNIST, Copyright (c) 2017 Zalando SE "
            "(https://tech.zalando.com), MIT licensed."
        ),
        source="github.com/zalandoresearch/fashion-mnist LICENSE",
    ),
    "kmnist": DatasetProvenance(
        license="CC-BY-SA-4.0",
        # CODH asks for this wording specifically; it is reproduced verbatim.
        attribution=(
            '"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji '
            'Dataset" (created by NIJL and others), doi:10.20676/00000341'
        ),
        source="github.com/rois-codh/kmnist README",
    ),
    "n_mnist": DatasetProvenance(
        license="CC-BY-SA-4.0",
        attribution=(
            "N-MNIST: Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N. "
            '"Converting Static Image Datasets to Spiking Neuromorphic '
            'Datasets Using Saccades", Frontiers in Neuroscience, vol.9, '
            "no.437, Oct. 2015."
        ),
        source="garrickorchard.com/datasets/n-mnist",
    ),
    "dvs128_gesture": DatasetProvenance(
        license="CC-BY-4.0",
        attribution=(
            "DVS128 Gesture (DvsGesture): Amir, A. et al. \"A Low Power, "
            'Fully Event-Based Gesture Recognition System", IEEE CVPR 2017.'
        ),
        source="IBM DvsGesture README (data.brainchip.com mirror)",
    ),
    "ssc": DatasetProvenance(
        license="CC-BY-4.0",
        attribution=(
            "Spiking Speech Commands: Cramer, B., Stradmann, Y., Schemmel, "
            "J., and Zenke, F. \"The Heidelberg Spiking Data Sets for the "
            'Systematic Evaluation of Spiking Neural Networks", IEEE TNNLS '
            "33, 2744-2757 (2022). doi:10.1109/TNNLS.2020.3044364"
        ),
        source="zenkelab.org/resources/spiking-heidelberg-datasets-shd",
    ),
    "qmnist": DatasetProvenance(
        license="BSD-3-Clause",
        attribution=(
            "QMNIST, Copyright (c) Facebook, Inc. and its affiliates, "
            "BSD-3-Clause. Yadav, C. and Bottou, L. \"Cold Case: The Lost "
            'MNIST Digits", NeurIPS 2019.'
        ),
        source="github.com/facebookresearch/qmnist LICENSE (3-clause BSD)",
    ),
    # --- this project's own synthetic data -------------------------------
    "sequence_toy": DatasetProvenance(
        license="BSD-3-Clause",
        attribution=(
            "Synthetic token-parity sequences generated by this project; no "
            "third-party data."
        ),
        source="generated in-repo by spikeforge.data.sequence_source",
    ),
    # --- no licence this project can record -------------------------------
    # These carry the marker rather than a licence, but for two different
    # reasons, and `source` says which. Most were checked and the publisher
    # simply states no licence -- a verified absence, which the marker
    # represents honestly and a guessed SPDX id would not. MNIST alone is
    # unchecked, because its page will not serve. Either way: confirm before
    # publishing a reference entry trained on one of them.
    "mnist": DatasetProvenance(
        license=UNVERIFIED,
        attribution=(
            "The MNIST database of handwritten digits, Yann LeCun, Corinna "
            "Cortes, and Christopher J.C. Burges."
        ),
        source=(
            "UNCHECKED: yann.lecun.com/exdb/mnist refused connection on "
            "every attempt (2026-09-14, 2026-09-15). The original "
            "distribution is widely reported to state no explicit licence "
            "and the cited CC-BY-SA-3.0 is secondary, so neither is "
            "asserted here. Retry from another network."
        ),
    ),
    "cifar10_dvs": DatasetProvenance(
        license=UNVERIFIED,
        attribution=(
            "CIFAR10-DVS: Li, H., Liu, H., Ji, X., Li, G., and Shi, L. "
            '"CIFAR10-DVS: An Event-Stream Dataset for Object '
            'Classification", Frontiers in Neuroscience 11:309 (2017).'
        ),
        source=(
            "UNCHECKED: figshare item 4724671 returned HTTP 403 and the "
            "dataset's download endpoints answer HTTP 202 with no body; the "
            "CC-BY-4.0 in circulation is secondary, so it is not asserted "
            "here. Retry from another network."
        ),
    ),
    "usps": DatasetProvenance(
        license=UNVERIFIED,
        attribution=(
            "USPS handwritten digits. Hull, J. J. \"A Database for "
            'Handwritten Text Recognition Research", IEEE PAMI 16(5), 1994. '
            "Distributed via the LIBSVM dataset archive."
        ),
        source=(
            "CHECKED, none stated: the LIBSVM multiclass archive torchvision "
            "downloads from publishes no licence or terms, crediting only "
            "'UCI, Statlog, StatLib and other collections'"
        ),
    ),
    "emnist_digits": DatasetProvenance(
        license=UNVERIFIED,
        attribution=(
            "EMNIST: Cohen, G., Afshar, S., Tapson, J., and van Schaik, A. "
            '"EMNIST: an extension of MNIST to handwritten letters", 2017. '
            "Derived from the NIST Special Database 19."
        ),
        source=(
            "CHECKED, none stated: NIST's own EMNIST page asks for the "
            "citation above but declares no licence or terms of use"
        ),
    ),
    "emnist_letters": DatasetProvenance(
        license=UNVERIFIED,
        attribution=(
            "EMNIST: Cohen, G., Afshar, S., Tapson, J., and van Schaik, A. "
            '"EMNIST: an extension of MNIST to handwritten letters", 2017. '
            "Derived from the NIST Special Database 19."
        ),
        source=(
            "CHECKED, none stated: NIST's own EMNIST page asks for the "
            "citation above but declares no licence or terms of use"
        ),
    ),
    "cifar10": DatasetProvenance(
        license=UNVERIFIED,
        attribution=(
            "CIFAR-10: Krizhevsky, A. \"Learning Multiple Layers of Features "
            'from Tiny Images", 2009.'
        ),
        source=(
            "CHECKED, none stated: the authors' own page asks that the tech "
            "report be cited but declares no licence or terms of use"
        ),
    ),
}


def dataset_provenance(name: str) -> DatasetProvenance:
    """Return one dataset's upstream licence and attribution.

    Raises ``KeyError`` for a dataset with no recorded provenance, rather
    than returning a permissive default: a dataset this project cannot
    describe is not one it should publish a checkpoint for.
    """
    return _PROVENANCE[name]


def provenance_names() -> Dict[str, DatasetProvenance]:
    """Return the whole table, for coverage checks and documentation."""
    return dict(_PROVENANCE)
