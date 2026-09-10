"""Inference on the displayed sample: prediction plus layer activity."""

import torch


def infer_spikes(net, spikes, num_classes, true_label=None, coding="rate",
                 hidden_cap=256):
    """Score one sample's encoded spikes and summarise its activity."""
    tracked = net.forward_spikes(spikes, track=True)
    logits = tracked["logits"][0].detach()
    probs = torch.softmax(logits, dim=0)
    totals = _class_totals(tracked["output"])
    payload = {
        "predicted": int(logits.argmax()),
        "confidence": float(probs.max()),
        "true_label": None if true_label is None else int(true_label),
        "class_spikes": [float(x) for x in totals],
        "output_over_time": _over_time(tracked["output"]),
        "coding": coding,
        "input_mode": coding,
        "num_steps": int(tracked["steps"]),
    }
    return _with_rasters(payload, tracked, num_classes, hidden_cap)


def _with_rasters(payload, tracked, num_classes, hidden_cap):
    """Attach bounded hidden/output rasters under a private payload key."""
    payload["_rasters"] = {
        "hidden": layer_raster(tracked["hidden"], hidden_cap),
        "output": layer_raster(tracked["output"], num_classes),
    }
    return payload


def layer_raster(frames, max_neurons):
    """Collapse tracked per-step [B,N] frames into a bounded raster."""
    matrix = torch.stack([frame[0] for frame in frames]).detach().cpu()
    matrix = matrix[:, :max(1, int(max_neurons))]
    time_idx, neuron_idx = torch.where(matrix > 0)
    return {
        "time": time_idx.tolist(),
        "neurons": neuron_idx.tolist(),
        "num_steps": int(matrix.size(0)),
        "num_neurons": int(matrix.size(1)),
    }


def _class_totals(output):
    """Sum output spikes over time into a per-class total."""
    stacked = torch.stack([frame[0] for frame in output])
    return stacked.sum(dim=0).detach()


def _over_time(output):
    """Return the [T][num_classes] output spike matrix for sample 0."""
    return [[float(x) for x in frame[0].detach()] for frame in output]
