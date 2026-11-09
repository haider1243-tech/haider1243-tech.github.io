"""PWA assets: the manifest and icons the phone's 'Add to Home Screen' needs."""
import json
import tomllib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_and_icons():
    m = json.loads((ROOT / "static" / "manifest.json").read_text())
    assert m["display"] == "standalone" and m["start_url"].startswith("/") and m["name"]
    sizes = {}
    for icon in m["icons"]:
        p = ROOT / icon["src"].replace("/app/static/", "static/")
        w, h = Image.open(p).size
        assert icon["sizes"] == f"{w}x{h}" and w == h
        sizes.setdefault(icon.get("purpose", "any"), []).append(w)
    assert 192 in sizes["any"] and 512 in sizes["any"] and 512 in sizes["maskable"]
    assert Image.open(ROOT / "static" / "apple-touch-icon.png").size == (180, 180)


def test_streamlit_config_serves_static_dark_theme():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["server"]["enableStaticServing"] is True
    assert cfg["theme"]["base"] == "dark"
    assert "address" not in cfg["server"] or cfg["server"]["address"] == "127.0.0.1"   # never public by default


def test_docker_image_ships_static_files():
    assert "COPY static ./static" in (ROOT / "Dockerfile").read_text()
