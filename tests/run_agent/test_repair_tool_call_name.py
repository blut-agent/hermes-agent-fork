"""Tests for AIAgent._repair_tool_call — tool-name normalization.

Regression guard for #14784: Claude-style models sometimes emit
class-like tool-call names (``TodoTool_tool``, ``Patch_tool``,
``BrowserClick_tool``, ``PatchTool``). Before the fix they returned
"Unknown tool" even though the target tool was registered under a
snake_case name. The repair routine now normalizes CamelCase,
strips trailing ``_tool`` / ``-tool`` / ``tool`` suffixes (up to
twice to handle double-tacked suffixes like ``TodoTool_tool``), and
falls back to fuzzy match.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

VALID = {
    "todo",
    "patch",
    "browser_click",
    "browser_navigate",
    "web_search",
    "read_file",
    "write_file",
    "terminal",
    "execute_code",
    "session_search",
}


@pytest.fixture
def repair():
    """Return a bound _repair_tool_call built on a minimal shell agent.

    We avoid constructing a real AIAgent (which pulls in credential
    resolution, session DB, etc.) because the repair routine only
    reads self.valid_tool_names. A SimpleNamespace stub is enough to
    bind the unbound function.
    """
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)

class TestExistingBehaviorStillWorks:
    """Pre-existing repairs must keep working (no regressions)."""

    def test_lowercase_already_matches(self, repair):
        assert repair("browser_click") == "browser_click"



class TestClassLikeEmissions:
    """Regression coverage for #14784 — CamelCase + _tool suffix variants."""

    def test_camel_case_no_suffix(self, repair):
        assert repair("BrowserClick") == "browser_click"



class TestEdgeCases:
    """Edge inputs that must not crash or produce surprising results."""

    def test_empty_string(self, repair):
        assert repair("") is None



class TestVolcEngineXmlPollution:
    """Regression coverage for #33007 — VolcEngine ``api/plan`` endpoint
    leaks raw XML attribute fragments into ``tool_use.name``.

    Observed in production with the ``anthropic_messages`` API mode:
        terminal" parameter="command" string="true
        execute_code" parameter="code" string="true
        session_search" parameter="session_id" string="true

    The fix trims at the first ``"``/``'``/``<``/``>`` so the rest of
    the repair pipeline can resolve the cleaned name to a real tool.
    """

    def test_terminal_with_xml_attribute_pollution(self, repair):
        # Exact pattern from the bug report (terminal call).
        polluted = 'terminal" parameter="command" string="true'
        assert repair(polluted) == "terminal"


    def test_tool_name_with_trailing_quote_only(self, repair):
        # Minimal leak — just a stray trailing quote, no full attribute.
        assert repair('terminal"') == "terminal"


    def test_clean_tool_name_unaffected_by_sanitizer(self, repair):
        # Pure passthrough — no XML/quote chars, no change.
        assert repair("execute_code") == "execute_code"
        assert repair("session_search") == "session_search"

    def test_space_separated_name_still_normalizes(self, repair):
        # Critical: the XML strip must NOT consume whitespace, or the
        # legitimate "write file" -> write_file repair path breaks.
        assert repair("write file") == "write_file"

    def test_leading_quote_falls_through_to_fuzzy_match(self, repair):
        # Sanitizer only trims when the XML char is at idx > 0 — a
        # name that *starts* with a quote is left untouched so the
        # rest of the pipeline (fuzzy match at 0.7 cutoff) can still
        # recover the obvious target.
        assert repair('"terminal"') == "terminal"


class TestGatedToolNotFuzzyRemapped:
    """Regression coverage for #94506 — check_fn-gated tools must not be
    fuzzy-matched onto a sibling tool. A gated tool (e.g. kanban_list
    hidden from kanban workers) is a real tool that's unavailable this
    turn; fuzzy-matching it onto a different tool (kanban_link) silently
    remaps a read onto a write.
    """

    @pytest.fixture
    def repair_gated(self):
        """Repair stub where kanban_list is gated (not in valid_tool_names)
        but exists in the full registry alongside kanban_link.
        """
        from run_agent import AIAgent

        # kanban_list is gated — absent from valid_tool_names
        valid_without_gated = {"kanban_link", "kanban_complete", "web_search"}
        stub = SimpleNamespace(valid_tool_names=valid_without_gated)

        # The full registry contains both kanban_list and kanban_link
        all_names = ["kanban_link", "kanban_list", "kanban_complete", "web_search"]

        with patch("tools.registry.registry.get_all_tool_names", return_value=all_names):
            yield AIAgent._repair_tool_call.__get__(stub, AIAgent)

    def test_gated_tool_returns_none_not_sibling(self, repair_gated):
        # kanban_list is a real tool gated by check_fn — must NOT
        # fuzzy-match to kanban_link (which is close at cutoff=0.7).
        assert repair_gated("kanban_list") is None

    def test_gated_tool_camel_case_returns_none(self, repair_gated):
        # CamelCase variant of a gated tool also must not remap.
        assert repair_gated("KanbanList") is None

    def test_truly_unknown_tool_still_fuzzy_matches(self, repair_gated):
        # A completely unknown name can still fuzzy-match to a valid tool.
        # "kanban_link" is close enough to "kanban_lnk" (typo).
        result = repair_gated("kanban_lnk")
        assert result == "kanban_link"
