"""Dashboard Jev key flow, using a fake SDK client (no network, no credentials)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.registry import ModelRegistry

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
SECRET = "sk-test-SECRET-123"


@pytest.fixture(scope="module", autouse=True)
def need_artifacts():
    try:
        ModelRegistry.load()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"trained artifacts not loadable here ({exc}); run scripts/train.py + benchmark.py first")


class FakeClient:
    calls = 0
    choice = "deep"

    def __init__(self, **kw):
        pass

    def system_one(self, state, questions):
        FakeClient.calls += 1
        assert "fast_probability" in state
        return type("R", (), {"choices": {"tier": type("C", (), {"choice": FakeClient.choice})()}})()


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    import typesafe_sdk
    FakeClient.calls, FakeClient.choice = 0, "deep"
    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", FakeClient)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


def run_app(page="Single transaction"):
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.radio[0].set_value(page).run()
    return at


def enable_jev(at, key=SECRET):
    at.sidebar.text_input(key="jev_key_input").set_value(key)
    at.sidebar.toggle(key="jev_on").set_value(True).run()


def test_without_key_jev_policy_uses_local_fallback():
    at = run_app()
    at.selectbox(key="single_policy").set_value("jev").run()
    assert not at.exception
    assert any("not enabled" in i.value for i in at.info)
    assert FakeClient.calls == 0


def test_key_then_button_calls_jev_once_and_safety_applies():
    at = run_app()
    enable_jev(at)
    at.selectbox(key="single_policy").set_value("jev").run()
    assert FakeClient.calls == 0                       # moving widgets never calls Jev
    at.button(key="jev_route_btn").click().run()
    assert FakeClient.calls == 1 and not at.exception
    assert any("`jev`" in m.value for m in at.markdown)
    at.slider[0].set_value(5).run()                    # editing inputs after routing must not call Jev again
    assert FakeClient.calls == 1


def test_key_is_never_rendered():
    at = run_app()
    enable_jev(at)
    at.selectbox(key="single_policy").set_value("jev").run()
    at.button(key="jev_route_btn").click().run()
    at.button(key="jev_test_btn").click().run()
    rendered = " ".join([m.value for m in at.markdown] + [str(j.value) for j in at.json] + [e.value for e in at.error] + [s.value for s in at.success])
    assert SECRET not in rendered


def test_test_button_and_clear_key():
    at = run_app()
    enable_jev(at)
    at.button(key="jev_test_btn").click().run()
    assert FakeClient.calls == 1 and any("deep" in s.value for s in at.success)
    at.button(key="jev_clear_btn").click().run()
    assert at.sidebar.text_input(key="jev_key_input").value == "" and at.sidebar.toggle(key="jev_on").value is False


def test_jev_page_mini_benchmark_fake_client():
    at = run_app("Jev")
    enable_jev(at)
    at.slider(key="jev_bench_n").set_value(10).run()
    at.button(key="jev_bench_btn").click().run()
    assert not at.exception and FakeClient.calls == 10
    assert "jev_bench" in at.session_state and at.session_state["jev_bench"]["jev_answered"] == 10
