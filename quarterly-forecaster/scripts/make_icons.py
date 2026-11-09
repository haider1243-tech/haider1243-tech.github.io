"""Generate the PWA icons in static/ (run once; output is committed)."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "static"
BG, BAND, LINE = (13, 13, 13), (57, 135, 229, 70), (57, 135, 229, 255)


def icon(size: int, safe: float = 1.0) -> Image.Image:
    """Fan chart glyph: history line, then a widening P10-P90 band and a P50 line."""
    s = size
    img = Image.new("RGBA", (s, s), BG + (255,))
    layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    m = s * (1 - safe) / 2 + s * 0.14           # margin (bigger for maskable safe zone)
    w = s - 2 * m
    x = lambda f: m + f * w  # noqa: E731
    y = lambda f: m + (1 - f) * w  # noqa: E731
    hist = [(x(0.0), y(0.30)), (x(0.12), y(0.38)), (x(0.22), y(0.33)), (x(0.32), y(0.46)), (x(0.45), y(0.42))]
    d.polygon([hist[-1], (x(1.0), y(0.92)), (x(1.0), y(0.18))], fill=BAND)
    lw = max(2, int(s * 0.035))
    d.line(hist, fill=LINE, width=lw, joint="curve")
    d.line([hist[-1], (x(1.0), y(0.60))], fill=LINE, width=lw)
    r = lw * 1.2
    d.ellipse([hist[-1][0] - r, hist[-1][1] - r, hist[-1][0] + r, hist[-1][1] + r], fill=LINE)
    img.alpha_composite(layer)
    return img.convert("RGB")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    icon(192).save(OUT / "icon-192.png", optimize=True)
    icon(512).save(OUT / "icon-512.png", optimize=True)
    icon(512, safe=0.8).save(OUT / "icon-maskable-512.png", optimize=True)
    icon(180).save(OUT / "apple-touch-icon.png", optimize=True)
    icon(256).save(OUT / "forecaster.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])  # shortcuts
    print("icons written to", OUT)
