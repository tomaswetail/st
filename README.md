# st

Historical football data and Stryktipset tooling.

Application Python packages live under `src/` (`calc`, `data_sources`, `objects`, `utils`, `scripts`, plus `database.py` / `main.py`). First-party imports are `src.`-prefixed absolute imports (`from src.calc... import ...`); the repository root is the sole path root, so run everything from the repo root with `python -m` and no `PYTHONPATH`.

API-Football results + SofaScore xG: [docs/football_data_ingestion.md](docs/product/football_data_ingestion.md)
