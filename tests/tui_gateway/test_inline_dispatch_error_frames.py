"""An inline handler's exception must become an error frame, never escape dispatch().

entry.py's stdio reader loop calls dispatch() unguarded; the long-handler pool path converts
handler exceptions to ``-32000`` error frames (server.dispatch's run()), and ws.py additionally
catches dispatch crashes for the WebSocket transport. The inline path had no such conversion:
since 026e3e84ea made _profile_home() fail closed (raise FileNotFoundError for an unknown
profile), a ``config.get`` with an unknown profile killed the whole TUI/desktop stdio backend
instead of answering the single request. These tests pin the contract at the dispatch boundary.
"""
import json

import pytest


def test_inline_handler_exception_becomes_error_frame(monkeypatch):
    from tui_gateway import server

    monkeypatch.setitem(server._methods, "crashy.inline", lambda rid, params: (_ for _ in ()).throw(
        FileNotFoundError("Profile 'does-not-exist' does not exist.")))
    resp = server.dispatch({"jsonrpc": "2.0", "id": 7, "method": "crashy.inline", "params": {}})
    assert resp is not None
    assert resp["error"]["code"] == -32000
    assert "does-not-exist" in resp["error"]["message"]
    assert resp["id"] == 7


def test_unknown_profile_config_get_survives_dispatch(tmp_path, monkeypatch):
    """Real-import E2E through dispatch(): the exact shape that killed the stdio backend.

    The autouse home-isolation fixture has already pointed HERMES_HOME at this test's
    tmp dir; seed the config file that profile scoping reads and ask dispatch() for a
    profile that does not exist. The response must be an error frame — not a raise.
    """
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text("model:\n  provider: custom\n")
    from tui_gateway import server

    resp = server.dispatch({"jsonrpc": "2.0", "id": 1,
                            "method": "config.get", "params": {"profile": "no-such-profile", "key": "full"}})
    assert resp is not None and "error" in resp, resp
    assert resp["error"]["code"] == -32000
    # json-serializable frame (it must survive write_json on the stdio transport)
    json.dumps(resp)
