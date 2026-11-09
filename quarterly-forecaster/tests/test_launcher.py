import os
import sys
import time

import pytest

from forecaster import launcher as L


def test_services_bind_loopback_only(tmp_path):
    svcs = {s.name: s.argv for s in L.build_services(tmp_path, python="py")}
    assert set(svcs) == {"monitor", "webhook", "dashboard"}
    assert svcs["monitor"] == ["py", "-m", "forecaster.monitor"]
    w = svcs["webhook"]
    assert w[w.index("--host") + 1] == "127.0.0.1" and "--trust-proxy" not in w
    d = svcs["dashboard"]
    assert d[d.index("--server.address") + 1] == "127.0.0.1" and d[d.index("--server.headless") + 1] == "true"
    assert "0.0.0.0" not in " ".join(sum(svcs.values(), []))


def test_tunnel_config_rewrites_for_local_run(tmp_path, monkeypatch):
    monkeypatch.setattr(L.shutil, "which", lambda name: None)
    (tmp_path / "cloudflared").mkdir()
    cfg = tmp_path / "cloudflared" / "config.yml"
    assert L.tunnel_config(tmp_path, tmp_path / "data") is None                 # not set up
    cfg.write_text("tunnel: abc\ncredentials-file: /etc/cloudflared/abc.json\ningress:\n"
                   "  - hostname: tv.example.com\n    path: ^/tv-alert$\n    service: http://webhook:8000\n"
                   "  - service: http_status:404\n")
    assert L.tunnel_config(tmp_path, tmp_path / "data") is None                 # program not installed
    monkeypatch.setattr(L.shutil, "which", lambda name: "/usr/bin/cloudflared")
    out = L.tunnel_config(tmp_path, tmp_path / "data").read_text()
    assert "service: http://127.0.0.1:8000" in out and "webhook:8000" not in out
    assert f"credentials-file: {(tmp_path / 'cloudflared' / 'abc.json').as_posix()}" in out
    assert "path: ^/tv-alert$" in out and "http_status:404" in out             # ingress unchanged
    svcs = {s.name: s.argv for s in L.build_services(tmp_path, python="py", tunnel=tmp_path / "t.yml")}
    assert "--trust-proxy" in svcs["webhook"] and svcs["cloudflared"][-1] == "run"


def test_check_config(tmp_path, monkeypatch):
    monkeypatch.setattr(L.shutil, "which", lambda name: None)
    for k in ("APP_PASSWORD_HASH", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "WEBHOOK_SECRET", "FRED_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    errs, _ = L.check_config(tmp_path)
    assert errs and ".env is missing" in errs[0]
    (tmp_path / ".env").write_text("FRED_API_KEY=x\n")
    errs, warns = L.check_config(tmp_path)
    assert "No dashboard password" in errs[0]
    assert any("Telegram" in w for w in warns) and any("WEBHOOK_SECRET" in w for w in warns)
    L.set_env_value(tmp_path / ".env", "APP_PASSWORD_HASH", "pbkdf2_sha256:1:aa:bb")
    assert L.check_config(tmp_path)[0] == []
    (tmp_path / ".env").write_text("FRED_API_KEY=x\n")
    monkeypatch.setenv("APP_PASSWORD_HASH", "pbkdf2_sha256:1:aa:bb")          # Docker-style env also counts
    assert L.check_config(tmp_path)[0] == []


def test_set_env_value(tmp_path):
    p = tmp_path / ".env"
    p.write_text("A=1\n# APP_PASSWORD_HASH=\nB=2\n")
    L.set_env_value(p, "APP_PASSWORD_HASH", "h:1")
    L.set_env_value(p, "B", "3")
    L.set_env_value(p, "C", "4")
    assert p.read_text() == "A=1\nAPP_PASSWORD_HASH=h:1\nB=3\nC=4\n"


def test_single_instance_lock(tmp_path):
    pid = tmp_path / "launcher.pid"
    assert L.acquire(pid) and L.running_pid(pid) == os.getpid()
    assert L.acquire(pid)                                   # same process may re-acquire
    pid.write_text("999999999")                             # stale pid from a crash
    assert L.running_pid(pid) is None and L.acquire(pid)


def test_supervisor_restarts_with_backoff_and_stops(tmp_path):
    t = [0.0]
    crash = L.Service("crash", [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    sleeper = L.Service("sleeper", [sys.executable, "-c", "import time; time.sleep(60)"])
    sup = L.Supervisor([crash, sleeper], tmp_path / "logs", tmp_path, env=dict(os.environ),
                       clock=lambda: t[0], echo=lambda m: None)
    sup.poll()
    first = sleeper.proc.pid
    crash.proc.wait(10)
    sup.poll()                                              # noticed the exit: scheduled in 2 s
    assert crash.proc is None and crash.restarts == 1 and crash.next_start == 2.0 and crash.backoff == 4.0
    t[0] = 2.0
    sup.poll()
    assert crash.proc is not None and sleeper.proc.pid == first
    crash.proc.wait(10)
    sup.poll()
    assert crash.next_start == 6.0 and crash.backoff == 8.0   # 2, 4, 8 ... capped at 5 min
    sup.stop()
    assert sleeper.proc.poll() is not None
    log = (tmp_path / "logs" / "crash.console.log").read_text()
    assert log.count("boom") == 2 and "restart #1" in log


def test_backoff_resets_after_stable_run(tmp_path):
    t = [0.0]
    s = L.Service("x", [sys.executable, "-c", "import sys; sys.exit(1)"])
    sup = L.Supervisor([s], tmp_path, tmp_path, env=dict(os.environ), clock=lambda: t[0], echo=lambda m: None)
    s.backoff = 256.0
    sup.poll()
    s.proc.wait(10)
    t[0] = 700.0                                            # ran > 10 min before exiting
    sup.poll()
    assert s.next_start == 702.0
    sup.stop()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal path")
def test_stop_running_terminates_process(tmp_path):
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = tmp_path / "launcher.pid"
    pid.write_text(str(p.pid))
    assert L.stop_running(pid, timeout=10)
    p.wait(5)
    assert not pid.exists() and not L.stop_running(pid)


def test_windows_scripts():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    start = (root / "start.bat").read_bytes().decode()
    setup = (root / "setup.bat").read_bytes().decode()
    ps1 = (root / "scripts" / "windows" / "create_shortcuts.ps1").read_bytes().decode()
    for txt in (start, setup, ps1):
        assert "\r\n" in txt and "\n" not in txt.replace("\r\n", "")     # CRLF only (cmd.exe needs it)
    assert "-m forecaster.launcher --open" in start and "/boot" in start and "setup.bat" in start
    assert "forecaster.auth --write-env" in setup and "create_shortcuts.ps1" in setup
    assert "tailscale serve --bg 8501" in setup and "funnel" not in setup.lower()   # tailnet only, never public
    assert "AtStartup" in ps1 and "ExecutionTimeLimit ([TimeSpan]::Zero)" in ps1 and "GetFolderPath(\"Startup\")" in ps1
    assert "$args" not in ps1.lower()                                       # reserved PowerShell variable
    assert (root / "static" / "forecaster.ico").exists()
