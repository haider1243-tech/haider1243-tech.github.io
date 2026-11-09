"""Dashboard password: PBKDF2-SHA256 hashes, constant-time checks, login lockouts.

Generate the hash for .env (the password itself is never stored):
    python -m forecaster.auth          # prompts twice, prints APP_PASSWORD_HASH=...

The app fails closed: with no APP_PASSWORD_HASH it renders nothing but a setup message.
Lockouts: 5 wrong passwords in one browser session lock that session for 5 minutes,
and 20 failures across all sessions within 15 minutes lock logins for everyone for
15 minutes. Failures are stored in data/auth_failures.json, so restarts don't reset them.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

ALGO = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 600_000      # OWASP 2023 guidance for PBKDF2-SHA256


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS, salt: bytes | None = None) -> str:
    if not password:
        raise ValueError("empty password")
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    b64 = lambda b: base64.b64encode(b).decode().rstrip("=")  # noqa: E731
    # ':' separators, not '$': Docker Compose expands $VAR inside env files and would mangle the hash
    return f"{ALGO}:{iterations}:{b64(salt)}:{b64(dk)}"


def _b64d(s: str) -> bytes:
    return base64.b64decode(s + "=" * (-len(s) % 4))


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time check; malformed or missing hashes never verify."""
    if not password or not stored:
        return False
    try:
        algo, iters, salt, expected = stored.strip().split(":")
        if algo != ALGO or int(iters) < 1000:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), _b64d(salt), int(iters))
        return hmac.compare_digest(dk, _b64d(expected))
    except (ValueError, TypeError):
        return False


class LoginGuard:
    """Global failure counter persisted to disk (survives restarts)."""

    def __init__(self, path: Path, max_failures: int = 20, window_s: float = 900.0, clock=time.time) -> None:
        self.path, self.max_failures, self.window_s, self.clock = Path(path), max_failures, window_s, clock

    def _load(self) -> list[float]:
        try:
            return [float(t) for t in json.loads(self.path.read_text())]
        except (FileNotFoundError, ValueError, TypeError):
            return []

    def _recent(self) -> list[float]:
        now = self.clock()
        return [t for t in self._load() if now - t < self.window_s]

    def locked_for(self) -> float:
        """Seconds until logins reopen (0 if open)."""
        recent = sorted(self._recent())
        if len(recent) < self.max_failures:
            return 0.0
        return max(0.0, recent[-self.max_failures] + self.window_s - self.clock())

    def record_failure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._recent() + [self.clock()]))
        tmp.replace(self.path)


def password_hash_from_env() -> str | None:
    v = os.environ.get("APP_PASSWORD_HASH", "").strip()
    return v or None


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - interactive
    import argparse
    import getpass

    ap = argparse.ArgumentParser(description="Create the dashboard password hash")
    ap.add_argument("--write-env", action="store_true", help="save it straight into .env")
    args = ap.parse_args(argv)
    p1 = getpass.getpass("Dashboard password (12+ characters, typing is hidden): ")
    p2 = getpass.getpass("Again: ")
    if p1 != p2:
        print("Passwords don't match.")
        return 1
    if len(p1) < 12:
        print("Use at least 12 characters.")
        return 1
    h = hash_password(p1)
    if args.write_env:
        from forecaster.config import PROJECT_ROOT
        from forecaster.launcher import set_env_value

        set_env_value(PROJECT_ROOT / ".env", "APP_PASSWORD_HASH", h)
        print("Saved to .env (only a hash; the password itself is not stored).")
        return 0
    print("Add this line to .env (the password itself is not stored):")
    print(f"APP_PASSWORD_HASH={h}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
