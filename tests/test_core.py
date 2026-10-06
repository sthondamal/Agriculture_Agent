import pandas as pd

from agriculture_rag.core import (
    recommend_crops,
    retrieve_evidence,
    validate_crops,
    validate_prices,
)


def sample_data():
    prices = validate_prices(pd.read_csv("data/sample_market_prices.csv"))
    crops = validate_crops(pd.read_csv("data/crop_profiles.csv"))
    return prices, crops


def test_sample_data_loads_and_recommendations_respect_filters():
    prices, crops = sample_data()

    recommendations = recommend_crops(prices, crops, region="Kolar", season="Rabi (Oct-Feb)")

    assert not recommendations.empty
    assert set(recommendations["season"]) == {"Rabi (Oct-Feb)"}
    assert all("Kolar" in regions for regions in crops.loc[crops["crop"].isin(recommendations["crop"]), "regions"])
    assert recommendations.iloc[0]["estimated_margin_inr_per_acre"] >= recommendations.iloc[-1]["estimated_margin_inr_per_acre"]


def test_hybrid_retrieval_returns_local_price_evidence():
    prices, crops = sample_data()

    evidence = retrieve_evidence("tomato modal price", prices, crops, region="Kolar", top_k=4)

    assert evidence
    assert any("Kolar" in item.text and "Tomato" in item.text for item in evidence)


def test_price_upload_requires_core_columns():
    try:
        validate_prices(pd.DataFrame({"date": ["2025-01-01"]}))
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("An incomplete price table should be rejected")
