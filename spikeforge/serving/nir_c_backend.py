"""The ``nir-c`` inference backend: serve a bundle through C.

The bundle is exported once, through the same
:meth:`DeploymentBundle.export_c_package
<spikeforge.serving.bundle_types.DeploymentBundle.export_c_package>`
a torch-free target consumes — so a topology the C runtime rejects
fails here with the same typed :class:`CExportError` naming its node
kinds — and the resulting graph runs in-process via
:class:`~spikeforge.serving.c_runtime_lib.NirCLibrary`, one frame per
step, with LI and Delay state carrying across steps inside C.

The C runtime is single-sample and CPU-only, so frames keep the
leading batch dimension the torch path requires but the batch must be
exactly 1, and inputs cross through CPU: ``device`` only places the
returned readout. Per-stage spikes and state capture are unavailable
— the C API exposes the readout alone — so ``step`` reports no
spikes and state methods refuse with :class:`StateError` rather than
inventing either.
"""

import os
import tempfile
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import torch

from spikeforge.serving.c_package import GRAPH_FILENAME
from spikeforge.serving.c_runtime_lib import NirCLibrary
from spikeforge.serving.errors import (
    BackendError,
    CRuntimeExecutionError,
    StateError,
)
from spikeforge.topology.spec import TopologySpec

if TYPE_CHECKING:
    from spikeforge.serving.bundle_types import DeploymentBundle


def _single_sample(frame: torch.Tensor) -> torch.Tensor:
    """Return ``frame``'s sample, refusing a bad dtype or batch."""
    if frame.dtype != torch.float32:
        raise ValueError(
            f"the nir-c backend needs float32 frames, not {frame.dtype}"
        )
    if frame.dim() < 2:
        raise ValueError(
            "the nir-c backend needs a [B, ...] frame, "
            f"not a {frame.dim()}D frame"
        )
    if frame.size(0) != 1:
        raise ValueError(
            "the nir-c backend steps one sample at a time, not "
            f"a batch of {frame.size(0)}"
        )
    return frame[0]


def _export_graph_json(bundle: "DeploymentBundle") -> str:
    """Return ``bundle``'s C-package graph as JSON text.

    The package is written to a temporary directory and only its
    graph file is read back, so the backend consumes byte for byte
    what a torch-free target would load.
    """
    with tempfile.TemporaryDirectory(prefix="spikeforge-c-") as path:
        bundle.export_c_package(path)
        joined = os.path.join(path, GRAPH_FILENAME)
        with open(joined, encoding="utf-8") as handle:
            return handle.read()


class NirCBackend:
    """Step a bundle's NIR graph through the C runtime, one frame each."""

    def __init__(
        self,
        graph: str,
        spec: TopologySpec,
        device: Union[str, torch.device] = "cpu",
        lib: Optional[str] = None,
    ) -> None:
        """Remember ``graph``; the library loads lazily on first step.

        Deferring the load keeps construction honest: backend misuse
        (a bad batch, a closed backend) reports before the library is
        ever touched, and a missing library fails on first use with
        the typed error explaining how to build it.
        """
        self._graph_text = graph
        self._spec = spec
        self._device = torch.device(device)
        self._lib_path = lib
        self._lib: Optional[NirCLibrary] = None
        self._closed = False

    @classmethod
    def from_bundle(
        cls,
        bundle: "DeploymentBundle",
        device: Union[str, torch.device] = "cpu",
        lib: Optional[str] = None,
    ) -> "NirCBackend":
        """Export ``bundle`` and return the backend running its graph."""
        return cls(_export_graph_json(bundle), bundle.spec, device, lib)

    @property
    def spec(self) -> TopologySpec:
        """Return the topology this backend executes."""
        return self._spec

    def reset(self) -> None:
        """Re-initialize the C runtime state, keeping the graph."""
        if self._closed:
            raise BackendError("this nir-c backend is closed")
        if self._lib is not None:
            self._lib.reset()

    def step(
        self, frame: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance one timestep; return ``(readout, {})``.

        The readout keeps its batch dimension; the spikes mapping is
        always empty because the C API only exposes the readout.
        """
        unbatched = self._unbatched(frame)
        library = self._ensure_loaded()
        values, shape = library.run_frame(
            unbatched.data_ptr(), tuple(unbatched.shape)
        )
        readout = torch.as_tensor(
            values, dtype=torch.float32, device=self._device
        ).reshape((1,) + tuple(shape))
        return readout, {}

    def capture_state(self) -> Dict[str, Any]:
        """Refuse: native C state has no serializable form."""
        raise StateError(
            "the nir-c backend cannot capture its native runtime state"
        )

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Refuse: native C state has no serializable form."""
        raise StateError(
            "the nir-c backend cannot restore a captured state"
        )

    def close(self) -> None:
        """Release the C graph and runtime; the backend stays closed."""
        if self._lib is not None:
            self._lib.close()
            self._lib = None
        self._closed = True

    def _ensure_loaded(self) -> NirCLibrary:
        """Return the graph-loaded library, loading it on first step."""
        if self._closed:
            raise BackendError("this nir-c backend is closed")
        if self._lib is None:
            library = NirCLibrary(self._lib_path)
            try:
                library.load_graph(self._graph_text)
            except CRuntimeExecutionError:
                library.close()
                raise
            self._lib = library
        return self._lib

    def _unbatched(self, frame: torch.Tensor) -> torch.Tensor:
        """Return ``frame``'s single sample on CPU, validating the batch."""
        data = _single_sample(frame)
        if data.dim() > 3:
            raise ValueError(
                "the nir-c backend needs 1 to 3 data dims, "
                f"not {data.dim()}"
            )
        return data.detach().to("cpu").contiguous()
