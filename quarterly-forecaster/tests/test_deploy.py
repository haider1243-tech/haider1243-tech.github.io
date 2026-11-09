"""Deployment invariants: private by default, only /tv-alert public, auto-restart, logs."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]


def test_services_present_and_restart():
    assert set(SERVICES) == {"dashboard", "webhook", "monitor", "cloudflared"}
    for name, s in SERVICES.items():
        assert s["restart"] == "unless-stopped", name
        assert s["logging"]["options"]["max-size"], name
        assert "no-new-privileges:true" in s["security_opt"], name
        if name != "cloudflared":
            assert s.get("init") is True, name             # signals reach the app; clean stops


def test_dashboard_bound_to_host_loopback_only():
    assert SERVICES["dashboard"]["ports"] == ["127.0.0.1:8501:8501"]
    for name in ("webhook", "monitor", "cloudflared"):
        assert "ports" not in SERVICES[name], f"{name} must not publish a host port"


def test_webhook_only_reachable_through_tunnel():
    w = SERVICES["webhook"]
    assert w.get("expose") == ["8000"] and "--trust-proxy" in w["command"]
    cf = yaml.safe_load((ROOT / "cloudflared" / "config.example.yml").read_text())
    ing = cf["ingress"]
    assert ing[0]["path"] == "^/tv-alert$" and ing[0]["service"] == "http://webhook:8000"
    assert ing[-1] == {"service": "http_status:404"}                # catch-all: everything else 404
    assert len(ing) == 2


def test_secrets_and_private_files_not_committed_or_baked():
    gi = (ROOT / ".gitignore").read_text().splitlines()
    for p in (".env", "portfolio.yaml", "reports/", "cloudflared/*.json", "cloudflared/config.yml", "logs/"):
        assert p in gi, p
    di = (ROOT / ".dockerignore").read_text().splitlines()
    for p in (".env", "portfolio.yaml", "cloudflared", "data", "reports"):
        assert p in di, p


def test_dockerfile_runs_as_non_root():
    df = (ROOT / "Dockerfile").read_text()
    assert "\nUSER app\n" in df and "useradd" in df


def test_volumes_and_logs():
    vols = SERVICES["monitor"]["volumes"]
    assert "./logs:/app/logs" in vols and "./data:/app/data" in vols and "./.env:/app/.env:ro" in vols
    assert SERVICES["dashboard"]["command"][:3] == ["bash", "scripts/run_logged.sh", "dashboard"]
    assert "--server.address" in SERVICES["dashboard"]["command"]


def test_pyproject_is_valid_for_pip_and_pytest():
    """Regression: filterwarnings once landed under [tool.setuptools.packages.find], which broke
    `pip install -e .` (and the Docker build) and silently disabled the pytest filters."""
    import tomllib
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert cfg["tool"]["setuptools"]["packages"]["find"] == {"where": ["src"]}
    assert cfg["tool"]["pytest"]["ini_options"]["filterwarnings"]
