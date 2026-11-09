# Running quarterly-forecaster 24/7, privately

What ends up reachable:

| Thing | Who can reach it | How |
|---|---|---|
| Dashboard (Streamlit) | Only your devices, signed in to your Tailscale account, and only with the dashboard password | Published on the host's `127.0.0.1:8501`, shared to your tailnet with `tailscale serve` |
| TradingView webhook | The internet, but **only** `POST https://<your-hostname>/tv-alert`, with the secret and rate limits | Cloudflare Tunnel (outbound-only; no open ports on your router or VPS) |
| Monitor + Telegram `/status` | Only your Telegram chat | Outbound long-polling; no inbound port |
| Everything else (`/`, `/health`, `/docs`, other paths, other hostnames) | Nobody | The tunnel returns 404 at Cloudflare's edge; the webhook container has no published port |

There is no public website. Don't run `tailscale funnel` (that makes things public), and don't open
port 8501 or 8000 in any firewall or router.

**Two ways to run it.** On a Windows home PC the simplest is the one-click launcher: double-click
`setup.bat` once, then `start.bat` / the desktop shortcut (see README, *Use it as a phone app*). It
runs the same services with the same privacy rules (loopback only, `tailscale serve`, tunnel only if
`cloudflared/config.yml` exists and `cloudflared` is installed). You can skip step 2 below; steps 3
and 4 (Tailscale, Cloudflare Tunnel) apply to both. For a VPS or Linux box, use Docker Compose as
described here.

---

## 0. Prerequisites

- A machine that stays on: your home PC (turn off sleep) or a small VPS (1-2 vCPU, 2 GB RAM is enough
  without FinBERT; 4 GB with it).
- Docker:
  - **Linux:** install [Docker Engine](https://docs.docker.com/engine/install/), then
    `sudo systemctl enable --now docker`, so it starts at boot.
  - **Windows / macOS:** Docker Desktop, with *Settings → General → Start Docker Desktop when you sign in*
    turned on. Also turn on automatic sign-in, or the stack only starts once you log in.
- A Telegram bot (token from @BotFather) and your chat ID (see README).
- **For the webhook only:** a domain whose DNS is managed by Cloudflare, on the free plan. A domain
  costs about $10–15/year if you don't have one. Cloudflare's free "quick tunnels" (`trycloudflare.com`)
  forward every path to a random URL that changes on restart, so they can't do the
  "only /tv-alert" rule. Use a named tunnel as below.

## 1. Get the code and configure it

```bash
git clone <your repo> && cd <repo>/quarterly-forecaster
cp .env.example .env
cp portfolio.example.yaml portfolio.yaml          # must exist before `docker compose up`
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # only to run the two helpers below
.venv/bin/python -m forecaster.auth               # prompts for a dashboard password, prints APP_PASSWORD_HASH=...
.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(32))"   # -> WEBHOOK_SECRET
```

Edit `.env`:

```
APP_PASSWORD_HASH=pbkdf2_sha256:600000:...        # from forecaster.auth (the password itself isn't stored)
WEBHOOK_SECRET=...                                # the random string; also goes in TradingView's alert JSON
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
FRED_API_KEY=...                                  # optional but recommended
FINNHUB_API_KEY=...                               # optional: news, earnings
```

`.env`, `portfolio.yaml`, `reports/`, `logs/` and `cloudflared/` credentials are all gitignored.

## 2. Start the private part (dashboard + monitor)

```bash
docker compose build
docker compose up -d dashboard monitor webhook
docker compose ps                     # all "running"; healthchecks turn "healthy" within a few minutes
curl -s http://127.0.0.1:8501/_stcore/health    # -> ok (on the machine itself)
```

Send `/status` to your bot. It should reply within a minute; the regime and forecast lines fill
in after the first daily refresh. To fill them now, run `docker compose exec monitor python -m forecaster.pipeline`.

## 3. Tailscale: the dashboard on your phone

1. Create a free account at [tailscale.com](https://tailscale.com) and install Tailscale on the
   **server** (home PC / VPS):
   - **Linux:** `curl -fsSL https://tailscale.com/install.sh | sh`, then `sudo tailscale up`.
   - **Windows / macOS:** install the app and sign in.
2. Install the **Tailscale app on your phone** and sign in with the same account. Both devices now
   appear under *Machines* in the [admin console](https://login.tailscale.com/admin/machines).
3. In the admin console, under *DNS*, make sure **MagicDNS** and **HTTPS Certificates** are enabled.
   `serve` asks you to enable HTTPS the first time if it's off.
4. On the server, share the dashboard **inside your tailnet only**:
   ```bash
   sudo tailscale serve --bg 8501       # proxies https://<machine>.<tailnet>.ts.net -> 127.0.0.1:8501
   tailscale serve status               # shows the URL, and that it's tailnet-only (not Funnel)
   ```
   `--bg` keeps it running across reboots. To undo: `sudo tailscale serve reset`.
5. On your phone, with Tailscale connected, open `https://<machine>.<tailnet>.ts.net` and log in with
   the dashboard password. Install it to the home screen (it's a PWA: own icon, full-screen):
   - **iPhone (Safari):** Share → *Add to Home Screen*.
   - **Android (Chrome):** ⋮ → *Install app* (or *Add to Home screen*).
   HTTPS from `tailscale serve` is what lets the phone install it; the icon still only works while
   Tailscale is connected, and the app still asks for the password.
6. Optional hardening: in the admin console, *Access controls*, allow only your own user to reach
   the server, and *Settings → Device approval* so new devices need your approval.

Off your tailnet, the URL doesn't resolve. The dashboard also refuses to render anything until the
password is entered. After 5 wrong passwords a browser session is locked for 5 minutes, and after
20 failures in 15 minutes all logins are locked for 15 minutes.

## 4. Cloudflare Tunnel: only `/tv-alert` is public

You need a domain on Cloudflare. In these steps it's `example.com`, and the webhook will live at
`tv.example.com`.

1. Install `cloudflared` on the server (or any computer you can copy files from):
   - **Linux (Debian/Ubuntu):** follow [Cloudflare's package instructions](https://pkg.cloudflare.com/).
   - **macOS:** `brew install cloudflared`. **Windows:** `winget install --id Cloudflare.cloudflared`.
2. Authorise it for your domain (opens a browser; pick `example.com`):
   ```bash
   cloudflared tunnel login                 # writes ~/.cloudflared/cert.pem
   ```
3. Create the tunnel and point the hostname at it:
   ```bash
   cloudflared tunnel create quarterly-forecaster          # prints the tunnel ID, writes ~/.cloudflared/<ID>.json
   cloudflared tunnel route dns quarterly-forecaster tv.example.com
   ```
4. Put the credentials and the config next to `docker-compose.yml`:
   ```bash
   cp ~/.cloudflared/<ID>.json cloudflared/
   cp cloudflared/config.example.yml cloudflared/config.yml
   # edit cloudflared/config.yml: replace both TUNNEL_ID with <ID>, and tv.example.com with your hostname
   ```
   The config has exactly two rules: `^/tv-alert$` on your hostname goes to the webhook container,
   and everything else gets `http_status:404`.
5. **Check the rules before going live.** cloudflared can tell you which rule each URL hits:
   ```bash
   cloudflared tunnel --config cloudflared/config.yml ingress validate
   cloudflared tunnel --config cloudflared/config.yml ingress rule https://tv.example.com/tv-alert   # -> rule #0, webhook
   cloudflared tunnel --config cloudflared/config.yml ingress rule https://tv.example.com/           # -> rule #1, 404
   cloudflared tunnel --config cloudflared/config.yml ingress rule https://tv.example.com/health     # -> rule #1, 404
   ```
   (Run these where cloudflared is installed; the credentials path in the file only matters when the
   tunnel actually runs.)
6. Start it:
   ```bash
   docker compose up -d cloudflared
   docker compose logs cloudflared | tail       # "Registered tunnel connection" x4
   ```
7. Test from outside (e.g. your phone on mobile data, Tailscale off):
   ```bash
   curl -i https://tv.example.com/                                  # 404
   curl -i https://tv.example.com/health                            # 404
   curl -i -X POST https://tv.example.com/tv-alert -d '{"secret":"wrong"}'    # 401
   ```
8. In TradingView, set the alert's webhook URL to `https://tv.example.com/tv-alert`, with the JSON
   message from the README and your `WEBHOOK_SECRET`.

Rate limits: 30 requests/minute per client IP and 120/minute in total, checked before the secret,
so guessing is slow. Change them with `WEBHOOK_RATE_PER_IP` / `WEBHOOK_RATE_TOTAL` in `.env`.

Optional, at Cloudflare's edge:
- *Security → WAF → Rate limiting rules* (the free plan includes one) on `/tv-alert`.
- A custom rule that blocks `/tv-alert` unless the source IP is one of TradingView's webhook IPs.
  Check [TradingView's current list](https://www.tradingview.com/support/solutions/43000529348-about-webhooks/)
  first; it can change.

## 5. Day to day

```bash
docker compose ps                         # status + health
docker compose logs -f monitor            # live logs (also in logs/monitor.log, webhook.log, dashboard.log)
docker compose restart dashboard          # after editing portfolio.yaml / watchlist.yaml
git pull && docker compose build && docker compose up -d   # update
docker compose down                       # stop everything (it won't restart until `up`)
```

- **Auto-restart:** every service has `restart: unless-stopped`. A crashed process is restarted
  within seconds, and after a reboot everything comes back once Docker starts. Only `docker compose down`
  or `stop` keeps it down.
- **Logs:** `logs/*.log` rotate at 5–10 MB, and Docker's own logs rotate at 10 MB × 5.
- **Schedules** run inside the `monitor` container in New York time:
  - checks every 15 minutes in market hours, hourly otherwise
  - data refresh Mon–Fri 06:15, and Saturday 06:15 with backtests
  - weekly report Sunday 07:23

  You don't need host cron. The old `scripts/*.sh` are for running without Docker.
- **Sentiment:** FinBERT isn't in the default image (it adds ~1.5 GB). For news sentiment, run
  `docker compose build --build-arg INSTALL_ML=true`.
- **Backups:** `data/` (SQLite logs, caches, snapshots), `reports/`, `.env`, `portfolio.yaml`, `cloudflared/`.
