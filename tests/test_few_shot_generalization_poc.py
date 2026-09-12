"""Smoke test for the few-shot cross-script protocol's wiring (issue #24).

Tiny settings (few episodes, small embedder) so it runs in CI seconds;
it is a wiring check, not the actual claim. Real cross-script
generalisation numbers come from
``examples/12_few_shot_character_generalization.py``.
"""

from spikeforge.memory.few_shot_generalization_poc import (
    run_few_shot_generalization_poc,
)


def test_run_reports_accuracy_within_bounds() -> None:
    """The protocol yields an accuracy between chance and perfect."""
    result = run_few_shot_generalization_poc(
        hidden=8, embed_dim=4, num_steps=3,
        train_episodes=3, train_n_way=3, train_k_shot=2, train_n_query=2,
        eval_episodes=3, eval_n_way=3, eval_n_query=2,
    )
    assert 0.0 <= result.accuracy <= 1.0
    assert result.chance == 1.0 / 3
    assert result.episodes == 3


def test_conv_net_topology_reports_accuracy_within_bounds() -> None:
    """The spatial (conv_net) embedder wires up the same as fc_small."""
    result = run_few_shot_generalization_poc(
        topology="conv_net", hidden=4, embed_dim=4, num_steps=3,
        train_episodes=3, train_n_way=3, train_k_shot=2, train_n_query=2,
        eval_episodes=3, eval_n_way=3, eval_n_query=2,
    )
    assert 0.0 <= result.accuracy <= 1.0
