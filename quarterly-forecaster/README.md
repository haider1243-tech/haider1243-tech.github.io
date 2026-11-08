# quarterly-forecaster

4-quarter-ahead probabilistic forecasts (P10 / P50 / P90) for SPY, QQQ, ^GSPC,
^IXIC, ^VIX and any tickers in `watchlist.yaml`. See [CLAUDE.md](CLAUDE.md) for
goals, design rules, and commands.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env   # add FRED_API_KEY / FINNHUB_API_KEY
pytest
python -m forecaster
```
