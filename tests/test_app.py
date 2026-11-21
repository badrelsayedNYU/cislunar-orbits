"""Smoke test: the Streamlit app runs end to end without exceptions."""

import pathlib

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = pathlib.Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_runs():
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.metric) >= 4


def test_app_manifolds():
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    at.checkbox[1].check().run()          # "Compute invariant manifolds"
    at.slider[1].set_value(10).run()      # few seeds to keep it fast
    assert not at.exception
