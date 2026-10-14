"""Host capabilities for C interop: the library path and libc's free.

The caller supplies the shared library: an explicit path wins, else
the ``NIR_C_RUNTIME_LIB`` environment variable. ``nir-c-runtime``
ships no installed ``.so`` itself, so the load error quotes the one
``cc`` command that builds one from its sources. The C runtime also
reports failures through malloc'd messages, released via
:func:`libc_free`.
"""

import ctypes.util
import os
from typing import Any, Optional

#: Environment variable naming the shared library, when no path is given.
LIB_ENV_VAR = "NIR_C_RUNTIME_LIB"

#: How to build the shared library; quoted by the load error.
BUILD_HINT = (
    "build it from the nir-c-runtime sources with "
    "`cc -shared -fPIC -O2 -std=c11 -I<nir-c-runtime>/include "
    "<nir-c-runtime>/src/*.c -o libnir_c_runtime.so -lm`, then pass "
    "its path explicitly or set NIR_C_RUNTIME_LIB"
)


def resolve_lib_path(explicit: Optional[str] = None) -> str:
    """Return the library path from ``explicit`` or the environment.

    An explicit path wins; otherwise ``NIR_C_RUNTIME_LIB`` is read.
    Neither is an error here — :class:`NirCLibrary
    <spikeforge.serving.c_runtime_lib.NirCLibrary>` raises the typed
    load error naming both — so callers can probe before committing.
    """
    if explicit is not None:
        return explicit
    return os.environ.get(LIB_ENV_VAR, "")


def libc_free() -> Any:
    """Return libc's ``free``, or ``None`` when libc cannot load.

    Without it the binding passes a NULL error pointer instead of
    leaking an unread message — terser failures, never a leak.
    """
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    libc = ctypes.CDLL(name)
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free.restype = None
    return libc.free
