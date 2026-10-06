# Workspace Instructions

- Keep market observations, crop assumptions, retrieval, and UI concerns separate.
- Treat bundled CSV data as synthetic demo data; never label it current, official, or live.
- Ground generated answers in retrieved records and keep price date, unit, and provenance visible.
- Keep crop margin calculations deterministic and label them estimates, not profit guarantees.
- Preserve an Ollama-free path so the app remains usable when no local model is running.
- Validate CSV schemas and add focused pytest coverage when changing retrieval or recommendation behavior.
- Run `python -m pytest -q` and `python -m compileall -q app.py agriculture_rag` after relevant changes.
