from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import requests
import streamlit as st

from agriculture_rag.core import recommend_crops, retrieve_evidence, validate_crops, validate_prices

ROOT = Path(__file__).parent
PRICE_PATH = ROOT / "data" / "sample_market_prices.csv"
CROP_PATH = ROOT / "data" / "crop_profiles.csv"
SEASONS = ["Any season", "Kharif (Jun-Sep)", "Rabi (Oct-Feb)"]
REGIONS = [
    "All Karnataka",
    "Kolar",
    "Mysuru",
    "Dharwad",
    "Belagavi",
    "Shivamogga",
    "Hassan",
    "Bengaluru Rural",
]

st.set_page_config(page_title="Karnataka Cropwise", page_icon="🌾", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#202a25; --muted:#69756e; --leaf:#2e6545; --lime:#b9cf79; --paper:#f5f4ed; --line:#dce2d7; --rust:#b65c3b; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
      .stApp { background: radial-gradient(ellipse at 90% 0%, #e7ecd8 0%, transparent 33%), linear-gradient(135deg,#f7f7f0,#eff2e8 62%,#f6f2e8); }
      h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:0 !important; }
      h1 { font-size:2.5rem !important; font-weight:800 !important; line-height:1.08 !important; }
      [data-testid="stSidebar"] { background:#e9eee3; border-right:1px solid var(--line); }
      [data-testid="stMetric"] { background:#fffefa; border:1px solid var(--line); border-radius:6px; padding:16px 18px; }
      [data-testid="stMetricLabel"] { color:var(--muted); }
      [data-testid="stMetricValue"] { font-family:'Manrope',sans-serif; }
      .eyebrow { color:var(--leaf); font-size:.75rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
      .hero { padding:1.5rem 0 1rem; border-bottom:1px solid var(--line); margin-bottom:1.1rem; }
      .hero p { color:var(--muted); max-width:700px; font-size:1.02rem; }
      .note { background:#fff6e6; border-left:4px solid #d29143; padding:12px 15px; border-radius:3px; font-size:.9rem; }
      .source { border-left:2px solid var(--lime); padding:7px 12px; margin:8px 0; background:#fbfcf8; color:#445148; font-size:.9rem; }
      div[data-testid="stTabs"] button { font-weight:600; }
      .stButton button[kind="primary"] { background:var(--leaf); border-color:var(--leaf); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_defaults() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        validate_prices(pd.read_csv(PRICE_PATH)),
        validate_crops(pd.read_csv(CROP_PATH)),
    )


def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"


def ollama_status(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def ask_ollama(base_url: str, model: str, question: str, evidence: list, recommendations: pd.DataFrame) -> str:
    evidence_text = "\n".join(f"- [{item.source}] {item.text}" for item in evidence)
    if recommendations.empty:
        recommendation_text = "No matching crop recommendation could be calculated from the loaded data."
    else:
        recommendation_text = recommendations.head(5).to_string(index=False)
    prompt = f"""You are a cautious Karnataka agricultural market assistant. Answer the farmer using only the supplied evidence. Clearly distinguish observed market data from modeled estimates. Never claim the illustrative sample data is current or official. Mention uncertainty, price date, unit, and relevant source names. Do not guarantee profit or give unsafe pesticide advice. If evidence is missing, say what data is needed.

Farmer question: {question}

Retrieved evidence:
{evidence_text}

Calculated crop margin estimates:
{recommendation_text}

Give a concise, practical answer and state that the margin estimate is gross sale value minus the listed per-acre cost, before unmodeled losses and price changes."""
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


with st.sidebar:
    st.markdown("<div class='eyebrow'>Your field context</div>", unsafe_allow_html=True)
    region = st.selectbox("Region / district", REGIONS)
    season = st.selectbox("Growing season", SEASONS, index=0)
    st.markdown("---")
    st.markdown("**Bring your own data**")
    price_upload = st.file_uploader("Market prices CSV", type=["csv"], key="prices")
    crop_upload = st.file_uploader("Crop assumptions CSV", type=["csv"], key="crops")
    st.caption("Price CSV: date, state, district, market, commodity, variety, min_price, modal_price, max_price, unit")
    st.caption("Crop CSV: crop, commodity, season, regions, expected_yield_q_per_acre, cost_inr_per_acre, duration_days, notes")
    st.markdown("---")
    st.markdown("**Local model**")
    ollama_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    default_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_model = st.text_input("Model tag", value=default_model)
    is_online = ollama_status(ollama_url)
    st.caption(("● Ollama available" if is_online else "○ Ollama not detected; grounded local response enabled"))

try:
    default_prices, default_crops = load_defaults()
    prices = validate_prices(pd.read_csv(price_upload)) if price_upload else default_prices.copy()
    crops = validate_crops(pd.read_csv(crop_upload)) if crop_upload else default_crops.copy()
except Exception as error:
    st.error(f"Could not load the supplied data: {error}")
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Karnataka · market intelligence for growers</div>
      <h1>Grow with a clearer<br>view of the market.</h1>
      <p>Explore observed mandi prices, compare crop-season estimates, and ask a locally grounded assistant what the numbers mean for your farm.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='note'><strong>Demo data notice:</strong> bundled market prices are illustrative sample records dated 15 Nov 2025, not live or official mandi quotes. Upload verified recent data before making a planting or selling decision. Yield and cost assumptions are also illustrative and should be replaced with local estimates.</div>",
    unsafe_allow_html=True,
)

latest_date = prices["date"].max().date().isoformat()
metric_one, metric_two, metric_three = st.columns(3)
metric_one.metric("Market records", f"{len(prices):,}")
metric_two.metric("Crop profiles", f"{len(crops):,}")
metric_three.metric("Latest record in data", latest_date)

prices_view = prices.copy()
if region != "All Karnataka":
    local = prices_view[
        prices_view["district"].str.contains(region, case=False, na=False)
        | prices_view["market"].str.contains(region, case=False, na=False)
    ]
    if not local.empty:
        prices_view = local

recommendations = recommend_crops(prices, crops, region=region, season=season)
prices_view = prices_view.sort_values("date", ascending=False)

active_view = st.radio(
    "Workspace",
    ["Market prices", "Crop & season planner", "Ask the assistant"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_view",
)
if active_view == "Market prices":
    st.subheader("Observed market prices")
    st.caption(f"Showing available records for {region}; prices are per listed unit. Data through {latest_date}.")
    search = st.text_input("Find a fruit, vegetable, or market", placeholder="Try tomato, onion, Kolar APMC")
    if search.strip():
        mask = prices_view.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
        prices_view = prices_view[mask]
    display_prices = prices_view[["date", "district", "market", "commodity", "variety", "min_price", "modal_price", "max_price", "unit"]].copy()
    display_prices["date"] = display_prices["date"].dt.date
    display_prices = display_prices.rename(columns={"date": "Date", "district": "District", "market": "Market", "commodity": "Commodity", "variety": "Variety", "min_price": "Min", "modal_price": "Modal", "max_price": "Max", "unit": "Unit"})
    st.dataframe(display_prices, use_container_width=True, hide_index=True)

if active_view == "Crop & season planner":
    st.subheader("Crop-season opportunity scan")
    st.caption("Illustrative contribution margin = latest matching modal price × expected yield − estimated per-acre cost. This is not a profit guarantee or complete farm budget.")
    if recommendations.empty:
        st.info("No crop profile has a matching market price for this region and season. Add market observations or broaden the region filter.")
    else:
        best = recommendations.iloc[0]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Top estimated margin", format_inr(best["estimated_margin_inr_per_acre"]))
        col_b.metric("Crop / season", f"{best['crop']} · {best['season']}")
        col_c.metric("Observed price date", best["price_date"])
        plan_display = recommendations.rename(columns={"crop": "Crop", "season": "Season", "observed_modal_price_inr_per_quintal": "Modal price / qtl (₹)", "price_date": "Price date", "expected_yield_q_per_acre": "Yield / acre (qtl)", "estimated_gross_inr_per_acre": "Gross / acre (₹)", "estimated_cost_inr_per_acre": "Cost / acre (₹)", "estimated_margin_inr_per_acre": "Estimated margin / acre (₹)", "duration_days": "Days", "assumptions": "Risk / planning note"})
        st.dataframe(plan_display, use_container_width=True, hide_index=True)
        st.caption("Prices and crop assumptions can change by variety, grade, water availability, labor, input prices, transport, and harvest timing. Validate with local agriculture officers and buyers.")

if active_view == "Ask the assistant":
    st.subheader("Ask about your market")
    question = st.text_area("Your question", placeholder="Which crop looks more promising for Kolar in the Rabi season, and what is the main risk?", height=90)
    if st.button("Find grounded answer", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving local evidence and calculating estimates…"):
            evidence = retrieve_evidence(question, prices, crops, region=region, season=season, top_k=6)
            try:
                if is_online:
                    answer = ask_ollama(ollama_url, ollama_model, question, evidence, recommendations)
                    st.markdown(f"**Local model · {ollama_model}**")
                else:
                    answer = "Ollama is not reachable, so here is the deterministic answer from the loaded records. "
                    if recommendations.empty:
                        answer += "There is not enough matching price and crop-profile data to rank a crop. Upload local market data and confirm your season."
                    else:
                        best = recommendations.iloc[0]
                        answer += (
                            f"The highest modeled margin in the selected filters is **{best['crop']}** for **{best['season']}**, "
                            f"about **{format_inr(best['estimated_margin_inr_per_acre'])} per acre** using a modal price of "
                            f"₹{best['observed_modal_price_inr_per_quintal']:,.0f}/quintal dated {best['price_date']}. "
                            f"Planning note: {best['assumptions']} This estimate is sensitive to yield, selling price, and costs."
                        )
                    answer += "\n\nThis is an estimate, not guaranteed profit."
                st.markdown(answer)
            except requests.RequestException as error:
                st.warning(f"Ollama request failed ({error}); showing the local data-only answer instead.")
                if recommendations.empty:
                    st.write("There is not enough matching price and crop-profile data to rank a crop.")
                else:
                    best = recommendations.iloc[0]
                    st.write(f"Current top estimate: {best['crop']} in {best['season']} at {format_inr(best['estimated_margin_inr_per_acre'])} per acre, using the {best['price_date']} observed price. Treat this as a planning estimate only.")
            st.markdown("**Retrieved evidence**")
            for item in evidence:
                st.markdown(f"<div class='source'><strong>{item.source}</strong><br>{item.text}</div>", unsafe_allow_html=True)
