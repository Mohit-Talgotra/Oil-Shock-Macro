"""Every page renders without an exception (Streamlit's AppTest)."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py"] + [f"pages/{p.name}" for p in sorted((APP / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES)
def test_missing_artifacts_show_friendly_message(page, tmp_path, monkeypatch):
    monkeypatch.setenv("OILSHOCK_ROOT", str(tmp_path))
    monkeypatch.delenv("OILSHOCK_DEMO", raising=False)
    at = st_testing.AppTest.from_file(str(APP / page), default_timeout=60).run()
    assert not at.exception


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_demo_mode_pages_render(page, tmp_path, monkeypatch):
    pytest.importorskip("statsmodels")
    monkeypatch.setenv("OILSHOCK_ROOT", str(tmp_path))
    monkeypatch.setenv("OILSHOCK_DEMO", "1")
    from oilshock import pipeline

    pipeline.main("all", synthetic_data=True, fast=True)
    at = st_testing.AppTest.from_file(str(APP / page), default_timeout=120).run()
    assert not at.exception
