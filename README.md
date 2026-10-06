# Karnataka Cropwise

A local-first Streamlit assistant for exploring Karnataka fruit and vegetable market records and comparing crop-season scenarios. It combines structured regional/season filters, TF-IDF retrieval, lexical matching with reciprocal-rank fusion, deterministic margin calculations, evidence snippets, and optional Ollama answer generation.

## Important data note

The bundled market CSV is synthetic illustrative data dated 15 November 2025. It is not a live price feed and is not an official government dataset. Crop yields and costs are also illustrative assumptions. Replace these CSVs with verified recent market observations and local agronomy/economic assumptions before relying on any result.

## Run

Python 3.11 or newer is recommended. From this project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The interface opens at the local Streamlit address printed in the terminal. Ollama is optional; the data-grounded deterministic response works when the model service is unavailable.

## Ollama (optional)

Install Ollama for Windows, start its local service, then download a model compatible with your machine. For example:

```powershell
ollama pull llama3.1:8b
```

The app defaults to `http://localhost:11434` and `llama3.1:8b`. Configure `OLLAMA_BASE_URL` and `OLLAMA_MODEL` environment variables or edit the sidebar fields. Requests and retrieved records stay local when using a local Ollama service.

## Data formats

Upload CSVs in the sidebar to replace the bundled files for the current session. Required columns:

- Market prices: `date,state,district,market,commodity,variety,min_price,modal_price,max_price,unit`
- Crop profiles: `crop,commodity,season,regions,expected_yield_q_per_acre,cost_inr_per_acre,duration_days,notes`

Prices are compared by matching commodity and the latest available price date. Crop margin is estimated as modal price per quintal multiplied by expected yield per acre, minus estimated per-acre cost. It excludes many real costs and risks, including price movement, grading, post-harvest loss, transport, financing, land rent, water constraints, and intercropping effects. Do not treat it as guaranteed profit or individualized agricultural advice.

## Retrieval and grounding

1. Filter candidate evidence by the selected region and growing season.
2. Score records using TF-IDF n-gram similarity and token overlap.
3. Fuse the two rankings using reciprocal-rank fusion.
4. Show source labels and the retrieved text alongside answers.
5. Use Ollama only to compose an answer from retrieved evidence and computed estimates; otherwise use a deterministic data-only response.

## Test

```powershell
python -m pytest -q
```

The included dataset is only a functional demo fixture. Verify all prices, varieties, units, yields, costs, and crop calendars with current local sources and qualified agricultural advisors before making decisions.
