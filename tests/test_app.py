"""Smoke tests for the Streamlit pilot platform.

These catch the failure mode that a manual check misses: Streamlit renders a
Python exception *inside* the page and still returns HTTP 200, so a broken app
looks healthy from the outside. ``AppTest`` runs the script headlessly and
surfaces the exception.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"
PAGES = ["Analyse a walk", "Trends", "Calibration", "Limitations"]


def _boot() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, f"app failed to start: {app.exception}"
    return app


def test_app_starts():
    app = _boot()
    assert app.sidebar.radio[0].options == PAGES


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(page):
    app = _boot()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, f"{page} raised: {app.exception}"


def test_pages_carry_the_screening_not_diagnostic_disclaimer():
    """The distinction is the whole framing of the tool; it must not go missing."""
    for page in ("Analyse a walk", "Trends"):
        app = _boot()
        app.sidebar.radio[0].set_value(page).run()
        captions = " ".join(c.value for c in app.caption)
        assert "not a diagnostic instrument" in captions, f"missing on {page}"


def test_limitations_page_reads_the_canonical_document():
    """Rendered from docs/limitations.md so app and repo cannot drift apart."""
    app = _boot()
    app.sidebar.radio[0].set_value("Limitations").run()

    body = " ".join(m.value for m in app.markdown)
    assert "Known limitations" in body
    assert "Frame rate limits stride-time variability" in body


def test_analyse_page_waits_for_an_upload():
    """With no video chosen the page must prompt, not attempt to analyse."""
    app = _boot()
    app.sidebar.radio[0].set_value("Analyse a walk").run()

    assert not app.exception
    assert any("Upload a video" in info.value for info in app.info)
