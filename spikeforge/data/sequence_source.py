"""A tiny synthetic token task that exercises the sequence vocabulary.

This is deliberately not a corpus pipeline: each sample is ``length`` integer
tokens drawn uniformly from ``[0, vocab)`` and the label is the token-sum
parity, a task the demonstration presets can separate. Long context, KV
caching, and large vocabularies are explicitly out of scope.
"""

from typing import Tuple

import torch

#: Demonstration task geometry shared with the sequence presets.
DEFAULT_LENGTH = 8
DEFAULT_VOCAB = 32
DEFAULT_FEATURES = 8
DEFAULT_CLASSES = 2


class SequenceSource:
    """Serve ``(tokens, label)`` pairs for the synthetic parity task."""

    def __init__(
        self,
        length: int = DEFAULT_LENGTH,
        vocab: int = DEFAULT_VOCAB,
        classes: int = DEFAULT_CLASSES,
        seed: int = 0,
    ) -> None:
        """Store the task geometry and seed its token generator."""
        self.length = int(length)
        self.vocab = int(vocab)
        self.classes = int(classes)
        self._generator = torch.Generator().manual_seed(int(seed))

    def tokens(self) -> torch.Tensor:
        """Return one ``[length]`` int64 sample of random tokens."""
        return torch.randint(
            0, self.vocab, (self.length,), generator=self._generator
        )

    def sample(self) -> Tuple[torch.Tensor, int]:
        """Return a ``(tokens, label)`` pair; the label is sum parity."""
        tokens = self.tokens()
        return tokens, int(tokens.sum().item() % self.classes)


def random_frames(
    steps: int,
    batch: int,
    length: int,
    features: int,
    seed: int = 0,
) -> torch.Tensor:
    """Return a deterministic ``[steps, batch, length, features]`` fixture."""
    generator = torch.Generator().manual_seed(int(seed))
    return torch.rand(
        steps, batch, length, features, generator=generator
    )
