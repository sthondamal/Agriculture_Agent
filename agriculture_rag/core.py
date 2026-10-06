from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRICE_COLUMNS = {
    "date",
    "state",
    "district",
    "market",
    "commodity",
    "variety",
    "min_price",
    "modal_price",
    "max_price",
    "unit",
}
CROP_COLUMNS = {
    "crop",
    "commodity",
    "season",
    "regions",
    "expected_yield_q_per_acre",
    "cost_inr_per_acre",
    "duration_days",
    "notes",
}


@dataclass(frozen=True)
class Evidence:
    source: str
    text: str
    score: float


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate uploaded market-price observations."""
    missing = PRICE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Price CSV is missing required columns: {', '.join(sorted(missing))}")

    cleaned = frame.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    for column in ("min_price", "modal_price", "max_price"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in ("state", "district", "market", "commodity", "variety", "unit"):
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()
    cleaned = cleaned.dropna(subset=["date", "modal_price"])
    cleaned = cleaned[cleaned["modal_price"] >= 0]
    if cleaned.empty:
        raise ValueError("Price CSV has no rows with a valid date and modal price.")
    return cleaned.reset_index(drop=True)


def validate_crops(frame: pd.DataFrame) -> pd.DataFrame:
    missing = CROP_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Crop CSV is missing required columns: {', '.join(sorted(missing))}")
    cleaned = frame.copy()
    for column in ("expected_yield_q_per_acre", "cost_inr_per_acre", "duration_days"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["expected_yield_q_per_acre", "cost_inr_per_acre", "duration_days"])
    cleaned = cleaned[(cleaned["expected_yield_q_per_acre"] > 0) & (cleaned["cost_inr_per_acre"] >= 0)]
    if cleaned.empty:
        raise ValueError("Crop CSV has no valid crop profiles.")
    return cleaned.reset_index(drop=True)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _price_text(row: pd.Series) -> str:
    return (
        f"{row['commodity']} {row['variety']} in {row['market']}, {row['district']}, Karnataka "
        f"on {row['date'].date()}: modal price Rs {row['modal_price']:.0f} per {row['unit']} "
        f"(range Rs {row['min_price']:.0f}-{row['max_price']:.0f})."
    )


def retrieve_evidence(
    query: str,
    prices: pd.DataFrame,
    crops: pd.DataFrame,
    region: str = "All Karnataka",
    season: str = "Any season",
    top_k: int = 5,
) -> list[Evidence]:
    """Hybrid retrieval: filtered TF-IDF similarity, token overlap, and RRF rank fusion."""
    records: list[tuple[str, str, str, str]] = []
    for _, row in prices.iterrows():
        text = _price_text(row)
        records.append((f"Market price: {row['market']} ({row['date'].date()})", text, _norm(row["district"]), "price"))
    for _, row in crops.iterrows():
        text = (
            f"{row['crop']} is a {row['season']} crop suited to {row['regions']}. "
            f"Expected yield {row['expected_yield_q_per_acre']} quintals per acre; estimated cost "
            f"Rs {row['cost_inr_per_acre']} per acre; duration {row['duration_days']} days. {row['notes']}"
        )
        records.append((f"Crop profile: {row['crop']}", text, _norm(row["regions"]), "crop"))

    if not records:
        return []
    normalized_region = _norm(region)
    normalized_season = _norm(season)
    query_text = f"{query} {region} {season}"
    query_tokens = set(re.findall(r"[a-z0-9]+", _norm(query_text)))
    documents = [record[1] for record in records]
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(documents + [query_text])
    semantic_scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

    eligible: list[int] = []
    for index, (source, text, location, kind) in enumerate(records):
        if normalized_region not in ("", "all karnataka", "karnataka"):
            location_match = normalized_region in location or normalized_region in _norm(text)
            if not location_match:
                continue
        if normalized_season not in ("", "any season") and kind == "crop":
            if normalized_season not in _norm(text):
                continue
        eligible.append(index)
    if not eligible:
        eligible = list(range(len(records)))

    lexical_scores = []
    for index in eligible:
        tokens = set(re.findall(r"[a-z0-9]+", _norm(documents[index])))
        lexical_scores.append(len(query_tokens & tokens) / max(1, len(query_tokens)))
    semantic_order = sorted(eligible, key=lambda index: semantic_scores[index], reverse=True)
    lexical_order = [eligible[position] for position in sorted(range(len(eligible)), key=lambda pos: lexical_scores[pos], reverse=True)]
    semantic_rank = {index: rank for rank, index in enumerate(semantic_order, start=1)}
    lexical_rank = {index: rank for rank, index in enumerate(lexical_order, start=1)}
    fused = sorted(
        eligible,
        key=lambda index: 1 / (60 + semantic_rank[index]) + 1 / (60 + lexical_rank[index]),
        reverse=True,
    )
    return [
        Evidence(records[index][0], records[index][1], float(semantic_scores[index]))
        for index in fused[: max(1, top_k)]
    ]


def recommend_crops(
    prices: pd.DataFrame,
    crops: pd.DataFrame,
    region: str = "All Karnataka",
    season: str = "Any season",
    top_k: int = 5,
) -> pd.DataFrame:
    """Estimate crop contribution margin from matching recent modal prices and profile assumptions."""
    candidates = crops.copy()
    if season != "Any season":
        candidates = candidates[candidates["season"].map(_norm) == _norm(season)]
    if region != "All Karnataka":
        candidates = candidates[candidates["regions"].map(lambda value: region.lower() in _norm(value))]

    market = prices.copy()
    if region != "All Karnataka":
        local = market[
            market["district"].map(lambda value: _norm(region) in _norm(value))
            | market["market"].map(lambda value: _norm(region) in _norm(value))
        ]
        if not local.empty:
            market = local

    output = []
    for _, crop in candidates.iterrows():
        matching = market[market["commodity"].map(_norm) == _norm(crop["commodity"])]
        if matching.empty:
            continue
        latest_date = matching["date"].max()
        latest = matching[matching["date"] == latest_date]
        modal_price = float(latest["modal_price"].median())
        yield_q = float(crop["expected_yield_q_per_acre"])
        cost = float(crop["cost_inr_per_acre"])
        gross = modal_price * yield_q
        output.append(
            {
                "crop": crop["crop"],
                "season": crop["season"],
                "price_commodity": crop["commodity"],
                "observed_modal_price_inr_per_quintal": modal_price,
                "price_date": latest_date.date().isoformat(),
                "expected_yield_q_per_acre": yield_q,
                "estimated_gross_inr_per_acre": gross,
                "estimated_cost_inr_per_acre": cost,
                "estimated_margin_inr_per_acre": gross - cost,
                "duration_days": int(crop["duration_days"]),
                "assumptions": crop["notes"],
            }
        )
    return pd.DataFrame(output).sort_values("estimated_margin_inr_per_acre", ascending=False).head(top_k).reset_index(drop=True) if output else pd.DataFrame()
