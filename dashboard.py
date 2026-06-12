"""
scripts/dashboard.py
====================
Optional interactive dashboard (Task C).

Launch with:
    streamlit run scripts/dashboard.py

Features
--------
- User search: enter any user ID and get personalised recommendations
- Movie explorer: see top-rated and most popular movies
- Model comparison: bar charts for RMSE & MAP@10
- Recommendation scores: inspect raw score distributions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Netflix Prize — Recommendation Dashboard",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data + model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_everything():
    from data_loader import get_dataset, train_test_split_temporal
    from recommender import Recommender
    from utils import load_model, load_results

    ratings, movies = get_dataset(sample_size=500_000)
    train, test = train_test_split_temporal(ratings, test_frac=0.2)

    # Try to load best available saved model
    for model_name in ["ncf", "svd", "als", "item_cf", "user_cf"]:
        try:
            model = load_model(model_name)
            break
        except FileNotFoundError:
            continue
    else:
        # Fallback: train a quick SVD
        from models import SVDModel
        model = SVDModel(n_factors=50, n_epochs=5)
        model.fit(train)
        model_name = "svd (quick)"

    rec = Recommender(model, movies=movies, ratings=ratings)

    try:
        results = load_results()
    except FileNotFoundError:
        results = {}

    return ratings, movies, rec, test, model_name, results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🎬 Netflix Recommender")
st.sidebar.markdown("Built on the **Netflix Prize Dataset**")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "👤 User Recommendations", "🎬 Movie Explorer", "📊 Model Comparison"],
)

try:
    ratings, movies, rec, test, model_name, results = load_everything()
    data_loaded = True
except Exception as e:
    st.error(f"Could not load data: {e}\n\nMake sure the dataset is downloaded and models are trained.")
    data_loaded = False
    st.stop()

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

if page == "🏠 Home":
    st.title("🎬 Netflix Prize Recommendation System")
    st.markdown(f"**Active model:** `{model_name}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings", f"{len(ratings):,}")
    col2.metric("Unique Users", f"{ratings['user_id'].nunique():,}")
    col3.metric("Unique Movies", f"{ratings['movie_id'].nunique():,}")
    col4.metric("Avg Rating", f"{ratings['rating'].mean():.2f} ★")

    st.markdown("---")
    st.subheader("Rating Distribution")
    counts = ratings["rating"].value_counts().sort_index().reset_index()
    counts.columns = ["Rating", "Count"]
    fig = px.bar(counts, x="Rating", y="Count", color="Rating",
                 color_continuous_scale="teal", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# User Recommendations
# ---------------------------------------------------------------------------

elif page == "👤 User Recommendations":
    st.title("👤 Personalised Recommendations")

    test_users = sorted(test["user_id"].unique().tolist())
    uid = st.selectbox("Select User ID", options=test_users[:500])
    top_k = st.slider("Number of Recommendations", 5, 20, 10)

    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations …"):
            recs_df = rec.recommend(int(uid), top_k=top_k)

        st.subheader(f"Top-{top_k} Recommendations for User {uid}")
        st.dataframe(recs_df, use_container_width=True)

        # History
        hist = rec.user_history(int(uid), n=10)
        if not hist.empty:
            st.subheader("User's Recent Ratings")
            cols = [c for c in ["title", "movie_id", "rating"] if c in hist.columns]
            st.dataframe(hist[cols], use_container_width=True)

        # Explain top recommendation
        if not recs_df.empty:
            movie_id = int(recs_df.iloc[0]["movie_id"])
            explanation = rec.explain(int(uid), movie_id)
            st.info(f"**Why '{recs_df.iloc[0]['title']}'?** {explanation}")

# ---------------------------------------------------------------------------
# Movie Explorer
# ---------------------------------------------------------------------------

elif page == "🎬 Movie Explorer":
    st.title("🎬 Movie Explorer")

    movie_stats = (
        ratings.groupby("movie_id")
        .agg(n_ratings=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
        .merge(movies[["movie_id", "title", "year"]], on="movie_id", how="left")
        .sort_values("n_ratings", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Popular Movies")
        top20 = movie_stats.head(20).copy()
        fig = px.bar(top20, x="n_ratings", y="title", orientation="h",
                     color="avg_rating", color_continuous_scale="teal",
                     template="plotly_dark")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Popularity vs Average Rating")
        fig2 = px.scatter(
            movie_stats,
            x="n_ratings", y="avg_rating",
            hover_data=["title", "year"],
            log_x=True,
            color="avg_rating",
            color_continuous_scale="teal",
            template="plotly_dark",
            opacity=0.5,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")

    if not results:
        st.warning("No saved evaluation results found. Run `scripts/train_models.py` first.")
    else:
        df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(df, use_container_width=True)

        metric = st.selectbox("Select Metric", options=[c for c in df.columns if c != "Model"])
        fig = px.bar(df, x="Model", y=metric, color="Model",
                     color_discrete_sequence=px.colors.sequential.Teal,
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Metric Definitions
        | Metric | Description |
        |--------|-------------|
        | **RMSE** | Root Mean Squared Error — measures rating prediction accuracy. Lower is better. |
        | **MAP@10** | Mean Average Precision at 10 — measures ranking quality. Higher is better. A movie is considered relevant if its rating ≥ 3.5. |
        | **MAE** | Mean Absolute Error — average absolute deviation of predicted ratings. |
        """)
