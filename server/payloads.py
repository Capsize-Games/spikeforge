"""Assemble outbound payloads from engines and encoder configs."""

from snn_interpreter import model_store


def compatibility(engine, encode):
    """Compare a checkpoint's training config to the encoder config."""
    expected = engine.input_mode
    current = encode.coding if encode is not None else "raw"
    dataset = encode.dataset if encode is not None else engine.dataset
    steps = encode.num_steps if encode is not None else engine.num_steps
    return {
        "dataset_match": engine.dataset == dataset,
        "coding_match": expected == current,
        "num_steps_match": int(engine.num_steps) == int(steps),
        "expected_input_mode": expected,
        "current_coding": current,
    }


def model_loaded_payload(engine, name, encode, accuracy):
    """Build the model_loaded payload including config coupling details."""
    return {
        "name": name,
        "dataset": engine.dataset,
        "accuracy": accuracy,
        "input_mode": engine.input_mode,
        "coding": engine.input_mode,
        "hidden": engine.net.hidden,
        "beta": engine.net.beta,
        "num_steps": engine.num_steps,
        "num_classes": engine.num_classes,
        "meta": _meta(name),
        "compatibility": compatibility(engine, encode),
    }


def _meta(name):
    """Read a checkpoint's stored meta, tolerating missing files."""
    try:
        return model_store.load(name).get("meta", {})
    except Exception:
        return {}
