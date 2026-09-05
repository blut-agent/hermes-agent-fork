"""Regression: a resumed session's stored title never raises Rich MarkupError (#103602).

``_preload_resumed_session`` renders the resume banner through ``_console_print`` (Rich markup
ON), and the title is DB content — the built-in ``/plan`` slash command literally stores
``[/plan — plan mode]``. Interpolated raw, that title closes the surrounding
``[{accent_color}]…`` block and Rich raises ``MarkupError`` before the input loop renders.
The startup ``_say`` path already escapes its title (same file, line ~459); this site was
missed. Contract: any stored title renders verbatim, never crashes.
"""

import pytest
from rich.errors import MarkupError

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _RecordingConsole:
    """Captures markup-ON prints and validates them the way Rich would."""

    def __init__(self):
        self.lines = []

    def print(self, text, *args, **kwargs):
        from rich.console import Console
        Console().print(text)  # raises MarkupError on unbalanced markup
        self.lines.append(text)


def _resume_cli(session_meta):
    from types import SimpleNamespace

    cli = SimpleNamespace(
        _resumed=True,
        _quiet_mode=False,
        session_id="20260905_101343_84feda",
        _session_db=SimpleNamespace(
            get_session=lambda _sid: session_meta,
            get_resume_conversations=lambda _sid: (
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            ),
            get_session_cwd=lambda _sid: None,
        ),
        console=_RecordingConsole(),
    )
    cli._restore_session_state = lambda *_a, **_k: None
    cli._reopen_session = lambda: None
    cli._follow_compression_chain = lambda meta, _notify: meta
    cli._resume_history_limit_error = lambda *a, **k: None
    cli._console_print = cli.console.print  # markup-ON console, like the live Rich console
    return cli


MARKUP_TITLE = "[/plan — plan mode]"


@pytest.mark.parametrize("title", [MARKUP_TITLE, "[error] from logs", "plain title"])
def test_resume_banner_renders_markup_titles_verbatim(title):
    cli = _resume_cli({"title": title})
    assert CLIAgentSetupMixin._preload_resumed_session(cli) is True
    banner = cli.console.lines[0]
    assert title in banner, f"title {title!r} must appear verbatim in {banner!r}"


def test_resume_banner_without_title_still_renders():
    cli = _resume_cli({"title": None})
    assert CLIAgentSetupMixin._preload_resumed_session(cli) is True
    assert cli.console.lines  # banner printed
