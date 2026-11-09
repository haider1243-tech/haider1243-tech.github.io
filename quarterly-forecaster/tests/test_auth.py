import warnings
from pathlib import Path

import pytest

from conftest import TEST_PASSWORD
from forecaster.auth import LoginGuard, hash_password, verify_password

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_hash_and_verify():
    h = hash_password("s3cret-pass", iterations=1000)
    assert h.startswith("pbkdf2_sha256:1000:") and "$" not in h     # '$' would break compose env files
    assert verify_password("s3cret-pass", h)
    assert not verify_password("s3cret-pasS", h)
    assert hash_password("x", iterations=1000) != hash_password("x", iterations=1000)   # salted


@pytest.mark.parametrize("stored", [None, "", "plaintext", "md5:1:aa:bb", "pbkdf2_sha256:10:aa:bb",
                                    "pbkdf2_sha256:abc:aa:bb", "pbkdf2_sha256:1000:!!:??"])
def test_malformed_hashes_never_verify(stored):
    assert not verify_password("anything", stored)


def test_login_guard_lockout(tmp_path):
    now = [1000.0]
    g = LoginGuard(tmp_path / "f.json", max_failures=3, window_s=60, clock=lambda: now[0])
    for _ in range(2):
        g.record_failure()
    assert g.locked_for() == 0
    g.record_failure()
    assert g.locked_for() == pytest.approx(60)
    now[0] += 61
    assert g.locked_for() == 0
    g2 = LoginGuard(tmp_path / "f.json", max_failures=3, window_s=60, clock=lambda: now[0])
    assert g2.locked_for() == 0                                     # persisted, old failures expired


def _app(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(tmp_path))
    return AppTest.from_file(APP, default_timeout=60)


def test_app_fails_closed_without_hash(tmp_path, monkeypatch):
    monkeypatch.delenv("APP_PASSWORD_HASH", raising=False)
    at = _app(tmp_path, monkeypatch).run()
    assert not at.exception
    assert at.error and "APP_PASSWORD_HASH" in at.error[0].value
    assert not at.sidebar.radio and not at.metric                  # nothing else rendered


def test_login_flow(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch).run()
    assert not at.sidebar.radio and at.text_input(key="login_pw")  # only the login form
    # form widgets: type and submit in the same run (as a browser does)
    at.text_input(key="login_pw").input("wrong")
    at.button[0].click().run()
    assert at.error and "Wrong password" in at.error[0].value and not at.sidebar.radio
    at.text_input(key="login_pw").input(TEST_PASSWORD)
    at.button[0].click().run()
    assert not at.exception and at.sidebar.radio                   # dashboard unlocked
    assert any("Backtest" in m.value for m in at.markdown) or at.info


def test_session_lockout_after_five_failures(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    at = _app(tmp_path, monkeypatch).run()
    for _ in range(5):
        at.text_input(key="login_pw").input("nope")
        at.button[0].click().run()
    at.text_input(key="login_pw").input(TEST_PASSWORD)
    at.button[0].click().run()
    assert at.error and "Too many failed attempts" in at.error[0].value
    assert not at.sidebar.radio                                    # even the right password is refused
