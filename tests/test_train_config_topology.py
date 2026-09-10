"""TrainConfig topology fields reach the built training engine."""

from server.schemas import TrainConfig
from server.training import TrainingService
from snn_interpreter.network.spiking_net import SpikingNet
from snn_interpreter.training.training_engine import TrainingEngine


def _engine(config: TrainConfig) -> TrainingEngine:
    """Build an engine from a config without touching any dataset."""
    return TrainingService._make_engine(config, None, None)


def test_train_config_defaults_to_fc_legacy() -> None:
    """The default config still selects the legacy topology."""
    config = TrainConfig(device="cpu")
    assert config.topology == "fc_legacy"
    assert config.topology_params == {}
    engine = _engine(config)
    assert engine.topology == "fc_legacy"
    assert isinstance(engine.net, SpikingNet)


def test_train_config_topology_reaches_engine() -> None:
    """The configured topology and its params reach the built engine."""
    config = TrainConfig(
        device="cpu",
        topology="conv_net",
        topology_params={"channels": 2},
    )
    engine = _engine(config)
    assert engine.topology == "conv_net"
    assert engine.spec.stage("conv1").params["out_channels"] == 2


def test_train_config_topology_params_override_defaults() -> None:
    """topology_params are validated, stored, and applied as overrides."""
    config = TrainConfig(
        device="cpu",
        topology="recurrent_net",
        topology_params={"hidden": 9, "beta": 0.25},
    )
    assert config.topology_params == {"hidden": 9, "beta": 0.25}
    engine = _engine(config)
    assert engine.topology == "recurrent_net"
    assert (engine.hidden, engine.beta) == (9, 0.25)
