"""Shared test doubles: one socket, engine, model, and ASGI driver."""

from _fakes.asgi import _get as _get
from _fakes.asgi import _get_json as _get_json
from _fakes.asgi import _http as _http
from _fakes.asgi import _http_json as _http_json
from _fakes.asgi import _post as _post
from _fakes.asgi import _post_json as _post_json
from _fakes.asgi import _websocket as _websocket
from _fakes.fake_model import FakeModel as FakeModel
from _fakes.fake_sample import FakeSample as FakeSample
from _fakes.fake_ws import FakeWS as FakeWS

__all__ = [
    "FakeModel",
    "FakeSample",
    "FakeWS",
    "_get",
    "_get_json",
    "_http",
    "_http_json",
    "_post",
    "_post_json",
    "_websocket",
]
