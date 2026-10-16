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
PAGES = ["Trends", "Analyse a walk", "Camera setup", "Limitations"]


def _boot() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, f"app failed to start: {app.exception}"
    return app


def test_app_starts():
    app = _boot()
    options = app.sidebar.radio[0].options
    assert [o.split("  ", 1)[-1] for o in options] == PAGES


def test_trends_is_the_landing_page():
    """The tool exists to follow people over time; that is what opens first."""
    app = _boot()
    assert app.sidebar.radio[0].value == "Trends"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(page):
    app = _boot()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, f"{page} raised: {app.exception}"


def test_pages_carry_the_screening_not_diagnostic_disclaimer():
    """The distinction is the whole framing of the tool; it must not go missing."""
    for page in ("Trends", "Analyse a walk", "Camera setup", "Limitations"):
        app = _boot()
        app.sidebar.radio[0].set_value(page).run()
        markdown = " ".join(m.value for m in app.markdown)
        assert "not a medical device" in markdown, f"missing on {page}"


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


# --------------------------------------------------------------------------
# with data in the database
# --------------------------------------------------------------------------
@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """Point the app at a temporary database holding a seeded history."""
    import streamlit as st

    import app.shared as shared
    from fixtures.history import seed_history
    from gaitscreen.config import Config
    from gaitscreen.storage.repository import SessionRepository

    db = tmp_path / "seeded.db"
    repository = SessionRepository(db)
    seed_history(repository)
    repository.close()

    cfg = Config.load().with_overrides({"storage": {"db_path": str(db)}})
    st.cache_resource.clear()
    monkeypatch.setattr(shared, "get_config", lambda: cfg)
    yield
    st.cache_resource.clear()


def _html(app) -> str:
    return " ".join(str(getattr(e, "proto", e)) for e in app.get("html"))


def test_trends_summarises_the_selected_person(seeded):
    app = _boot()
    assert not app.exception
    assert app.selectbox[0].options == ["Ada Example", "Ravi Example"]

    page = _html(app)
    assert "Worth discussing" in page          # the verdict for the latest walk
    assert "Walking speed" in page             # concerns and latest-walk tiles
    assert "Walks saved" in page


def test_the_camera_setup_is_not_listed_as_a_person(seeded):
    app = _boot()
    assert "__camera_setup__" not in app.selectbox[0].options


def test_switching_person_shows_their_own_summary(seeded):
    app = _boot()
    app.selectbox[0].set_value("Ravi Example").run()
    assert not app.exception
    assert "No concerns" in _html(app)


def test_analysis_is_open_once_the_camera_is_set_up(seeded):
    app = _boot()
    app.sidebar.radio[0].set_value("Analyse a walk").run()
    assert not app.exception
    assert "4.50 m" in _html(app)
    assert any("Upload a video to begin." == i.value for i in app.info)


def test_camera_setup_page_shows_the_saved_setup(seeded):
    app = _boot()
    app.sidebar.radio[0].set_value("Camera setup").run()
    assert not app.exception
    assert "4.50" in _html(app)


def test_samples_can_be_picked_without_a_camera_setup():
    """A visitor trying the tool has no camera setup, and must not need one:
    each sample carries the setup it was filmed with."""
    from app.samples import available_samples

    if not available_samples():
        pytest.skip("sample videos are not present in this checkout")
    app = _boot()
    app.sidebar.radio[0].set_value("Analyse a walk").run()
    app.session_state["video_source"] = "Try a sample"
    app.run()

    assert not app.exception
    titles = [r for r in app.radio if r.label == "Sample walk"]
    assert titles, "sample picker missing"
    assert "Comfortable pace" in titles[0].options[0]
    analyse_button = [b for b in app.button if b.label == "Analyse this walk"]
    assert analyse_button and not analyse_button[0].disabled
