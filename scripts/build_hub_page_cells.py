"""One table row per catalog entry, HTML-escaping every field."""

import html
from typing import Any, Dict

UNVERIFIED_CANDIDATE = "unverified-candidate"
#: The one catalog source whose entries carry trained weights.
TRAINED_SOURCE = "reference"
#: Raw-file base for the shipped weight bytes beside the catalog.


def _verified(entry: Dict[str, Any]) -> bool:
    """Return False only for the explicit unverified-candidate marker."""
    return entry.get("license") != UNVERIFIED_CANDIDATE


def _trained(entry: Dict[str, Any]) -> bool:
    """Return True only for an entry that carries trained weights."""
    return entry.get("source") == TRAINED_SOURCE


def _score_text(accuracy: Any, samples: Any, dataset: str) -> str:
    """Return the scored-samples line, or an empty string when unscored."""
    if accuracy is None:
        return ""
    return (
        f'<span class="score">{accuracy}%</span> on {samples} held-out '
        f"{dataset} samples"
    )


def _weights_cell(entry: Dict[str, Any]) -> str:
    """Return the cell that says whether this entry is a usable model.

    This is the column a visitor actually came for. A catalog of untrained
    shapes and a catalog of trained models are different products, so the
    page states which each row is instead of leaving it to the notes.
    Trained rows are identified without exposing direct artifact links.
    """
    if not _trained(entry):
        return (
            '<span class="badge badge-untrained">structure only</span>'
        )
    accuracy = entry.get("test_accuracy")
    samples = entry.get("test_samples")
    dataset = html.escape(str(entry.get("dataset", "")))
    score = _score_text(accuracy, samples, dataset)
    return (
        '<span class="badge badge-trained">trained weights</span><br>'
        f"{score}"
    )


def _license_badge(license_id: str) -> str:
    """Return the badge for a training-data licence id."""
    if license_id == UNVERIFIED_CANDIDATE:
        return (
            '<span class="badge badge-unverified">'
            f"{html.escape(license_id)}</span>"
        )
    return f"<code>{html.escape(license_id)}</code>"


def _dataset_cell(entry: Dict[str, Any]) -> str:
    """Return the cell naming the training data's own terms and credit.

    The `License` column beside this one describes the *weights* -- this
    project's artifact. This one describes the data they were trained on,
    which is a different licence with a different holder. Showing only the
    first invites a reader to assume it covers both.
    """
    dataset = entry.get("dataset")
    if not dataset:
        return '<span class="muted">not trained on a dataset</span>'
    license_id = str(entry.get("dataset_license") or "")
    badge = _license_badge(license_id)
    attribution = html.escape(str(entry.get("dataset_attribution") or ""))
    return (
        f"<strong>{html.escape(str(dataset))}</strong><br>{badge}<br>"
        f'<span class="attribution">{attribution}</span>'
    )


def _locator(entry: Dict[str, Any]) -> str:
    """Return the entry's locator: repo, URL, or topology name."""
    value = (
        entry.get("hf_repo") or entry.get("url")
        or entry.get("topology") or ""
    )
    return str(value)


def _row(entry: Dict[str, Any]) -> str:
    """Return one ``<tr>`` for ``entry``, HTML-escaping every field."""
    esc = html.escape
    verified = _verified(entry)
    badge_class = "badge-verified" if verified else "badge-unverified"
    badge_text = "verified source" if verified else "unverified candidate"
    locator = _locator(entry)
    return f"""<tr>
  <td><strong>{esc(str(entry.get("name", "")))}</strong><br>
    <code>{esc(str(entry.get("id", "")))}</code></td>
  <td>{_weights_cell(entry)}</td>
  <td>{esc(str(entry.get("framework", "")))}</td>
  <td>{esc(str(entry.get("kind", "")))}</td>
  <td>{esc(str(entry.get("source", "")))}<br>
    <code>{esc(str(locator))}</code></td>
  <td>{esc(str(entry.get("license", "")))}
    <span class="badge {badge_class}">{badge_text}</span></td>
  <td>{_dataset_cell(entry)}</td>
  <td>{esc(str(entry.get("notes", "")))}</td>
</tr>"""


def _trained_body(trained_count: int) -> str:
    """Return the notice body for a catalog with trained entries."""
    return (
        f"<strong>{trained_count} entries carry trained weights</strong> "
        "-- checkpoints this project trained itself, shipped with "
        "<code>spikeforge-hub</code>, each showing what it scores on the "
        "complete held-out test split. They are reference configurations "
        "with stock hyperparameters, <strong>not</strong> tuned attempts "
        "at state of the art, and every one names the command that "
        "reproduces it. The remaining entries are "
        "<code>bundled</code>: this project's own NIR graph presets -- "
        "structure, with freshly-initialised weights. No third-party "
        "weights are redistributed."
    )


def _metadata_body() -> str:
    """Return the notice body for a metadata-only catalog."""
    return (
        "This catalog is <strong>metadata-only</strong>: entries marked "
        "<code>bundled</code> are this project's own NIR graph presets "
        "(structure, not third-party trained weights); no third-party "
        "weights are redistributed."
    )


def _notice(trained_count: int) -> str:
    """Return the standing honesty notice, sized to what the catalog holds."""
    curation = (
        "https://github.com/Capsize-Games/spikeforge/blob/main/"
        "spikeforge_hub/CURATION.md"
    )
    body = (
        _trained_body(trained_count) if trained_count else _metadata_body()
    )
    return (
        f'<div class="notice">\n{body}\nSee\n'
        f'<a href="{curation}">CURATION.md</a>\n'
        "for the verification policy, and open a pull request there to "
        "propose a\nreal, checked entry.\n</div>"
    )
