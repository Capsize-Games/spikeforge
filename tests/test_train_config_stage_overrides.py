"""Additive ``TrainConfig`` stage overrides and the config->engine wiring."""

from server.schemas.train_config import TrainConfig
from server.training import TrainingService, _topology_params


def test_train_config_stage_defaults_are_empty() -> None:
    """The new fields are additive and default to empty mappings."""
    config = TrainConfig()
    assert config.stage_neurons == {}
    assert config.stage_params == {}


def test_train_config_accepts_stage_overrides() -> None:
    """The schema round-trips per-stage kind and param overrides."""
    config = TrainConfig(
        stage_neurons={"lif1": "synaptic"},
        stage_params={"lif1": {"reset": "zero"}},
    )
    dumped = config.model_dump()
    assert dumped["stage_neurons"] == {"lif1": "synaptic"}
    assert dumped["stage_params"] == {"lif1": {"reset": "zero"}}


def test_topology_params_fold_in_stage_overrides() -> None:
    """The engine hand-off folds the stage fields into topology params."""
    config = TrainConfig(
        topology_params={"hidden": 4},
        stage_neurons={"lif1": "synaptic"},
        stage_params={"lif2": {"beta": 0.5}},
    )
    params = _topology_params(config, {})
    assert params["hidden"] == 4
    assert params["neurons"] == {"lif1": "synaptic"}
    assert params["stage_params"] == {"lif2": {"beta": 0.5}}


def test_checkpoint_params_win_but_stage_fields_layer_on() -> None:
    """Stored checkpoint params win; the stage fields stay additive."""
    config = TrainConfig(
        topology_params={"hidden": 8},
        stage_neurons={"lif1": "synaptic"},
    )
    meta = {"topology_params": {"hidden": 2, "num_classes": 3}}
    params = _topology_params(config, meta)
    assert params["hidden"] == 2
    assert params["num_classes"] == 3
    assert params["neurons"] == {"lif1": "synaptic"}


def test_make_engine_honors_stage_neurons() -> None:
    """A config-to-engine build produces the requested heterogeneous spec."""
    config = TrainConfig(
        topology="fc_small",
        topology_params={"hidden": 4, "num_classes": 3},
        stage_neurons={"lif1": "synaptic"},
    )
    engine = TrainingService._make_engine(config, None, None)
    assert engine.spec.stage("lif1").kind == "synaptic"
    assert engine.spec.stage("lif2").kind == "leaky"
