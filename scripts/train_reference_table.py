"""The published reference configurations and where results land."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_OUT = REPO_ROOT / "build" / "reference"


@dataclass
class Reference:
    """One published reference configuration."""

    name: str
    dataset: str
    topology: str
    epochs: int
    num_steps: int
    #: The engine always injects ``hidden``/``beta`` into the preset's
    #: parameters, so a row that wants a preset's own defaults has to restate
    #: them here -- otherwise every fully-connected preset trains as the same
    #: network and the table silently compares a topology against itself.
    hidden: int = 128
    beta: float = 0.5
    batch_size: int = 128
    #: Divisor, not a count: ``subset=1`` trains on the whole training split.
    subset: int = 1
    lr: float = 1e-2
    seed: int = 0
    note: str = ""
    topology_params: Dict[str, Any] = field(default_factory=dict)

    def command(self) -> str:
        """Return the one-liner that reproduces this row."""
        return (
            "python scripts/train_reference_models.py "
            f"--only {self.name}"
        )

    def engine_kwargs(self) -> Dict[str, Any]:
        """Return the TrainingEngine arguments for this configuration."""
        return {
            "dataset": self.dataset,
            "topology": self.topology,
            "hidden": self.hidden,
            "beta": self.beta,
            "epochs": self.epochs,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "subset": self.subset,
            "lr": self.lr,
            "seed": self.seed,
            "device": "cpu",
            "topology_params": dict(self.topology_params),
        }


#: The published table. Kept small and honest on purpose: every row is a
#: shipped dataset on a shipped topology, trained from stock hyperparameters.
REFERENCES = (
    Reference(
        name="mnist-fc-legacy",
        dataset="mnist",
        topology="fc_legacy",
        epochs=3,
        num_steps=25,
        note="The default topology and the README quickstart's dataset.",
    ),
    Reference(
        name="mnist-fc-small",
        dataset="mnist",
        topology="fc_small",
        epochs=3,
        num_steps=25,
        hidden=32,
        beta=0.9,
        note="The smaller fully-connected preset, for edge-sized budgets.",
    ),
    Reference(
        name="mnist-conv-net",
        dataset="mnist",
        topology="conv_net",
        epochs=2,
        num_steps=25,
        note="Convolutional LIF; fewer epochs because each one costs more.",
    ),
    Reference(
        name="mnist-recurrent-net",
        dataset="mnist",
        topology="recurrent_net",
        epochs=3,
        num_steps=25,
        hidden=64,
        beta=0.9,
        note="Recurrent LIF with a one-step delayed feedback edge.",
    ),
    Reference(
        name="fashion-fc-legacy",
        dataset="fashion",
        topology="fc_legacy",
        epochs=3,
        num_steps=25,
        note="Fashion-MNIST: same shape, a harder ten-class problem.",
    ),
    Reference(
        name="dvs128-gesture-conv-net",
        dataset="dvs128_gesture",
        topology="conv_net",
        epochs=3,
        num_steps=25,
        # One bridged sample is 25 x 2 x 128 x 128 floats (~3.3 MB at fp32),
        # so the image rows' batch of 128 would be hundreds of megabytes of
        # input tensor alone before any activations.
        batch_size=16,
        # `num_classes` is deliberately absent: the engine injects the
        # registry's count (11), so stating it here would be a second place
        # for it to drift. `input_size` is sensor geometry, which the registry
        # does not carry -- a mismatch raises EventGeometryError by name.
        topology_params={"in_channels": 2, "input_size": 128},
        note=(
            "The event-native row: a real DVS recording rather than a "
            "re-rendered static image, on the canonical gesture task."
        ),
    ),
    Reference(
        name="ssc-fc-legacy",
        dataset="ssc",
        topology="fc_legacy",
        epochs=3,
        num_steps=25,
        batch_size=128,
        # A cochlea has 700 channels and no second spatial axis, so the flat
        # sensor area is 700. `num_classes` is absent on purpose: the engine
        # injects the registry's 35.
        topology_params={"input_size": 700},
        note=(
            "The audio row: a spiking cochlea model of Speech Commands, so "
            "the catalog spans static vision, event vision, and audio."
        ),
    ),
    Reference(
        name="kmnist-fc-legacy",
        dataset="kmnist",
        topology="fc_legacy",
        epochs=3,
        num_steps=25,
        note="Kuzushiji-MNIST: cursive Japanese characters.",
    ),
)

