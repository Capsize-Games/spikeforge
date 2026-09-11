"""Sparse event data model, dense accumulation, and the spike bridge.

This package is the event-modality counterpart to ``data`` plus
``encoding``: :class:`EventSample` carries an address-event stream, the
``dense`` helpers accumulate it into per-timestep frames, ``synthetic``
fabricates reproducible streams offline, and ``event_bridge`` converts a
stream into the spike tensor the simulator consumes.
"""
