"""One-command launcher: monitor + webhook + dashboard (+ Cloudflare tunnel), kept alive.

    python -m forecaster.launcher            # run everything in the foreground (Ctrl+C stops all)
    python -m forecaster.launcher --open     # ...and open the app in the browser once it's up
    python -m forecaster.launcher --stop     # stop a running launcher and its services
    python -m forecaster.launcher --status   # is it running? which services are configured?
    python -m forecaster.launcher --check    # configuration check only

start.bat / start.sh call this; setup.bat creates the desktop + Startup shortcuts.

Privacy is the same as the Docker deployment: the dashboard and webhook listen on
127.0.0.1 only. The phone reaches the dashboard through `tailscale serve` (tailnet
only, never `funnel`); TradingView reaches ONLY /tv-alert through cloudflared, if
cloudflared/config.yml exists and the `cloudflared` program is installed.

Each service's console output goes to logs/<name>.console.log (rotated at 10 MB);
the launcher itself logs to logs/launcher.log. A service that exits is
restarted with backoff (2 s doubling to 5 min; reset after 10 min of uptime). Only
one launcher runs at a time (data/launcher.pid).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from forecaster.config import PROJECT_ROOT

DASHBOARD_PORT = 8501
WEBHOOK_PORT = 8000
LOG_MAX_BYTES = 10 * 1024 * 1024
WINDOWS = os.name == "nt"


@dataclass
class Service:
    name: str
    argv: list[str]
    proc: subprocess.Popen | None = None
    started: float = 0.0
    backoff: float = 2.0
    next_start: float = 0.0
    restarts: int = 0
    log: object | None = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def read_env(path: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(path).items() if v} if path.exists() else {}


def set_env_value(path: Path, key: str, value: str) -> None:
    """Set KEY=value in a .env file (replacing an existing or commented-out line)."""
    lines = path.read_text().splitlines() if path.exists() else []
    pat = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=")
    out, done = [], False
    for ln in lines:
        if pat.match(ln) and not done:
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(ln)
    if not done:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n")


def tunnel_config(root: Path, runtime_dir: Path) -> Path | None:
    """cloudflared/config.yml rewritten for a non-Docker run, or None if not set up.

    In Docker the webhook is `http://webhook:8000` and credentials live in /etc/cloudflared;
    here it's 127.0.0.1 and the credentials file sits next to config.yml."""
    cfg = root / "cloudflared" / "config.yml"
    if not cfg.exists() or not shutil.which("cloudflared"):
        return None
    txt = cfg.read_text().replace("http://webhook:8000", f"http://127.0.0.1:{WEBHOOK_PORT}")

    def creds(m: re.Match) -> str:
        p = Path(m.group(2).strip().strip("'\""))
        if not p.exists():
            p = root / "cloudflared" / p.name
        return f"{m.group(1)}{p.as_posix()}"

    txt = re.sub(r"^(\s*credentials-file:\s*)(.+)$", creds, txt, flags=re.M)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    out = runtime_dir / "cloudflared.runtime.yml"
    out.write_text(txt)
    return out


def build_services(root: Path = PROJECT_ROOT, python: str = sys.executable,
                   tunnel: Path | None = None) -> list[Service]:
    """The commands to supervise. Everything binds to loopback."""
    webhook = [python, "-m", "forecaster.webhook_server", "--host", "127.0.0.1", "--port", str(WEBHOOK_PORT)]
    if tunnel is not None:
        webhook.append("--trust-proxy")           # only cloudflared can reach it, so its CF-Connecting-IP is real
    svcs = [
        Service("monitor", [python, "-m", "forecaster.monitor"]),
        Service("live", [python, "-m", "forecaster.live_feed"]),    # Finnhub + OANDA streams, only while running
        Service("shadow", [python, "-m", "forecaster.shadow"]),     # paper-trades lab setups on live bars; no orders
        Service("webhook", webhook),
        Service("dashboard", [python, "-m", "streamlit", "run", str(root / "app.py"),
                              "--server.address", "127.0.0.1", "--server.port", str(DASHBOARD_PORT),
                              "--server.headless", "true", "--browser.gatherUsageStats", "false"]),
    ]
    if tunnel is not None:
        svcs.append(Service("cloudflared", [shutil.which("cloudflared") or "cloudflared", "tunnel",
                                            "--no-autoupdate", "--config", str(tunnel), "run"]))
    return svcs


def check_config(root: Path = PROJECT_ROOT) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors stop the launcher."""
    errors, warns = [], []
    env_file = root / ".env"
    env = {**read_env(env_file), **{k: v for k, v in os.environ.items() if v}}
    if not env_file.exists():
        errors.append(".env is missing: run setup.bat (or copy .env.example to .env).")
    elif not env.get("APP_PASSWORD_HASH"):
        errors.append("No dashboard password: run setup.bat, or `python -m forecaster.auth --write-env`.")
    if not env.get("TELEGRAM_BOT_TOKEN") or not env.get("TELEGRAM_CHAT_ID"):
        warns.append("Telegram not configured: alerts are logged but not sent.")
    if not env.get("WEBHOOK_SECRET"):
        warns.append("WEBHOOK_SECRET not set: the TradingView webhook refuses every alert (503).")
    if not env.get("FRED_API_KEY"):
        warns.append("FRED_API_KEY not set: no macro data or CPI/PPI/jobs dates.")
    if (root / "cloudflared" / "config.yml").exists() and not shutil.which("cloudflared"):
        warns.append("cloudflared/config.yml exists but the cloudflared program isn't installed: "
                     "TradingView can't reach the webhook.")
    return errors, warns


# --------------------------------------------------------------------------- #
# Single instance
# --------------------------------------------------------------------------- #

def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if WINDOWS:  # pragma: no cover - Windows only
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                     # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_pid(pid_file: Path) -> int | None:
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def acquire(pid_file: Path) -> bool:
    other = running_pid(pid_file)
    if other and other != os.getpid():
        return False
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    return True


def stop_running(pid_file: Path, timeout: float = 20.0) -> bool:
    pid = running_pid(pid_file)
    if not pid:
        return False
    if WINDOWS:  # pragma: no cover - Windows only: kill the launcher and its whole process tree
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGTERM)
    end = time.time() + timeout
    while time.time() < end and pid_alive(pid):
        time.sleep(0.2)
    pid_file.unlink(missing_ok=True)
    return True


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #

def _open_log(log_dir: Path, name: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    f = log_dir / f"{name}.console.log"     # monitor/webhook also write their own <name>.log
    if f.exists() and f.stat().st_size > LOG_MAX_BYTES:
        os.replace(f, log_dir / f"{name}.console.log.1")
    return open(f, "ab", buffering=0)


class Supervisor:
    def __init__(self, services: list[Service], log_dir: Path, cwd: Path, env: dict | None = None,
                 clock=time.monotonic, max_backoff: float = 300.0, stable_after: float = 600.0,
                 echo=print) -> None:
        self.services = services
        self.log_dir, self.cwd = log_dir, cwd
        self.env = env if env is not None else self._child_env(cwd)
        self.clock, self.max_backoff, self.stable_after = clock, max_backoff, stable_after
        self.echo = echo
        self.stopping = False

    @staticmethod
    def _child_env(cwd: Path) -> dict:
        env = dict(os.environ)
        src = str(cwd / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _start(self, s: Service) -> None:
        if s.log is None:
            s.log = _open_log(self.log_dir, s.name)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        s.log.write(f"\n--- {stamp} launcher: starting {s.name} (restart #{s.restarts}) ---\n".encode())
        kw = {}
        if WINDOWS:  # pragma: no cover
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        s.proc = subprocess.Popen(s.argv, cwd=self.cwd, env=self.env, stdout=s.log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, **kw)
        s.started = self.clock()
        self.echo(f"started {s.name} (pid {s.proc.pid})")

    def poll(self) -> None:
        """Start anything not running whose backoff has passed. Call every second or so."""
        now = self.clock()
        for s in self.services:
            if s.proc is not None:
                code = s.proc.poll()
                if code is None:
                    if now - s.started > self.stable_after:
                        s.backoff = 2.0                       # healthy for a while: reset backoff
                    continue
                up = now - s.started
                if up > self.stable_after:
                    s.backoff = 2.0
                s.next_start = now + s.backoff
                self.echo(f"{s.name} exited (code {code}) after {up:.0f}s; restarting in {s.backoff:.0f}s")
                s.backoff = min(self.max_backoff, s.backoff * 2)
                s.proc = None
                s.restarts += 1
            if not self.stopping and now >= s.next_start:
                self._start(s)

    def stop(self, timeout: float = 15.0) -> None:
        self.stopping = True
        for s in self.services:
            if s.proc is not None and s.proc.poll() is None:
                s.proc.terminate()
        end = time.monotonic() + timeout
        for s in self.services:
            if s.proc is None:
                continue
            try:
                s.proc.wait(max(0.1, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                s.proc.kill()
                s.proc.wait()
            self.echo(f"stopped {s.name}")
        for s in self.services:
            if s.log is not None:
                s.log.close()
                s.log = None

    def run(self, stop_event=None, interval: float = 1.0) -> None:
        try:
            while not self.stopping and not (stop_event and stop_event()):
                self.poll()
                time.sleep(interval)
        finally:
            self.stop()


def wait_for_dashboard(url: str, timeout: float = 90.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - long-running
    ap = argparse.ArgumentParser(description="Start monitor + webhook + dashboard together")
    ap.add_argument("--open", action="store_true", help="open the dashboard in the browser once it's up")
    ap.add_argument("--stop", action="store_true", help="stop the running launcher")
    ap.add_argument("--status", action="store_true", help="show whether it's running")
    ap.add_argument("--check", action="store_true", help="check the configuration and exit")
    args = ap.parse_args(argv)
    root = PROJECT_ROOT
    pid_file = root / "data" / "launcher.pid"
    url = f"http://127.0.0.1:{DASHBOARD_PORT}"
    log_dir = root / "logs"

    if args.stop:
        print("Stopped." if stop_running(pid_file) else "Not running.")
        return 0
    errors, warns = check_config(root)
    if args.status or args.check:
        pid = running_pid(pid_file)
        print(f"Launcher: {'running (pid %d)' % pid if pid else 'not running'} · dashboard {url}")
        for w in warns:
            print("WARNING:", w)
        for e in errors:
            print("ERROR:", e)
        return 1 if errors else 0
    for w in warns:
        print("WARNING:", w)
    if errors:
        for e in errors:
            print("ERROR:", e)
        return 2
    if not acquire(pid_file):
        print(f"Already running. Dashboard: {url}")
        if args.open:
            import webbrowser
            webbrowser.open(url)
        return 0

    from forecaster.monitor import setup_logging

    setup_logging(log_dir, "launcher")
    log = logging.getLogger("launcher")
    tunnel = tunnel_config(root, root / "data")
    sup = Supervisor(build_services(root, tunnel=tunnel), log_dir, root, echo=log.info)
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    if WINDOWS and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)
    log.info("Quarterly Forecaster running. Dashboard: %s (logs in %s)", url, log_dir)
    if tunnel is None:
        log.info("Cloudflare tunnel: not configured (TradingView alerts can't reach this PC).")
    if args.open:
        import threading
        import webbrowser

        threading.Thread(target=lambda: wait_for_dashboard(url) and webbrowser.open(url), daemon=True).start()
    try:
        from forecaster import project_status
        from forecaster.config import load_settings

        project_status.start_background(load_settings().cache_dir, root)   # refreshes STATUS.md
    except Exception as exc:
        log.warning("status check not started: %s", exc)
    try:
        sup.run(stop_event=lambda: stop["flag"])
    finally:
        if running_pid(pid_file) == os.getpid():
            pid_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
