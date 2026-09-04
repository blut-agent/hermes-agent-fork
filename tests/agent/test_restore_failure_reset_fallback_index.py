# Copyright 2026 Nous Research (Licensed under the Apache License, Version 2.0)
"""Regression tests for the fallback-chain stranding half of #102860.

When the primary runtime restore fails partway (client rebuild raises, pool
rebind fails, a malformed snapshot key is missing, ...), ``restore_primary_
runtime()`` returned ``False`` without resetting ``_fallback_index``. The
mid-turn fallback walk had already advanced the index past the entries it
consumed, so every subsequent failure for the rest of the session resumed
the chain *past* the healthy early entries — e.g. a Bedrock → DeepSeek →
Regolo chain silently degraded to Bedrock → Regolo, with DeepSeek (a healthy
provider) never tried again. On a fresh process the first failed restore
poisoned the whole session the same way.

The fix resets ``_fallback_index = 0`` in the failure path so the next walk
starts from the top. The same-backend skip guard in ``try_activate_fallback()``
keeps the reset from re-activating the backend that just failed.

The gated early-return paths (rate-limit cooldown, primary-pool reset wait)
intentionally do NOT reset the index: staying gated is what prevents the
#24996 cross-turn chain-replay storm, and those tests still pass.
"""

from unittest.mock import MagicMock, patch

from agent.agent_init import _snapshot_primary_runtime
from run_agent import AIAgent


def _make_agent(fallback_model=None):
    """A real AIAgent (mocked transport) in the post-fallback state."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _snapshot(agent):
    """The construction-time primary snapshot via the production snapshotter."""
    _snapshot_primary_runtime(agent)
    return agent._primary_runtime


def _activate_fallback(agent):
    """Walk the chain one step with the transport resolver stubbed."""
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_client(), "resolved"),
    ):
        return agent._try_activate_fallback()


def _mock_client(base_url="https://fallback.example/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


class TestFailedRestoreResetsFallbackIndex:
    """A failed restore must not strand _fallback_index past consumed entries."""

    def test_failed_rebuild_resets_index(self):
        """Client rebuild raising mid-restore leaves _fallback_index at 0."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        agent._primary_runtime = _snapshot(agent)

        # Fall back (index → 1), then make the rebuild blow up like a Bedrock
        # restore hitting _create_openai_client() with no OPENAI_API_KEY.
        assert _activate_fallback(agent) is True
        assert agent._fallback_index == 1
        with patch.object(
            agent, "_create_openai_client", side_effect=RuntimeError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the OPENAI_API_KEY environment variable"
            )
        ):
            assert agent._restore_primary_runtime() is False
        assert agent._fallback_index == 0, (
            "failed restore stranded _fallback_index; the next failure would "
            "skip this chain entry for the rest of the session (#102860)"
        )

    def test_chain_retried_from_top_after_failed_restore(self):
        """The entry skipped by the stranded index is reachable on the next walk."""
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "deepseek", "model": "deepseek-chat"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent._primary_runtime = _snapshot(agent)

        assert _activate_fallback(agent) is True  # consumed entry 0 → index 1
        with patch.object(agent, "_create_openai_client", side_effect=RuntimeError("boom")):
            assert agent._restore_primary_runtime() is False
        # Before the fix the next walk resumed at index 1 (deepseek) and entry 0
        # was never tried again this session. It must be first in line again.
        assert agent._fallback_chain[agent._fallback_index] == fbs[0]

    def test_successful_restore_still_resets_index(self):
        """The fix adds a reset; the success path's own reset must keep working."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        agent._primary_runtime = _snapshot(agent)

        assert _activate_fallback(agent) is True
        assert agent._restore_primary_runtime() is True
        assert agent._fallback_activated is False
        assert agent._fallback_index == 0


class TestGatedRestoresKeepIndex:
    """Gated early returns intentionally keep the index (#24996 replay guard)."""

    def test_rate_limit_gated_restore_does_not_reset_index(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        agent._primary_runtime = _snapshot(agent)

        assert _activate_fallback(agent) is True
        assert agent._fallback_index == 1
        agent._rate_limited_until = float("inf")  # primary in cooldown
        assert agent._restore_primary_runtime() is False
        assert agent._fallback_index == 1, (
            "gated restore must keep the index: resetting would let a client "
            "replaying every turn walk the whole chain again (#24996)"
        )
