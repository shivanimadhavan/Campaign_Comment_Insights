"""WEFRA Campaign Comment Insights — Streamlit dashboard.
Run from the project root: python -m streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import select, func

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import (  # noqa: E402
    SessionLocal, Product, Campaign, Post, Comment, CommentAnalysis, FDAReaction
)

st.set_page_config(page_title="Campaign Comment Insights", page_icon="💬", layout="wide")
st.title("💬 Campaign Comment Insights")
st.caption("WEFRA LIFE · Halvoria Pharma | Local SQLite dashboard")

@st.cache_data(ttl=30)
def fetch_data():
    with SessionLocal() as db:
        rows = db.execute(
            select(
                Comment.comment_id.label("comment_id"),
                Comment.comment_text.label("comment_text"),
                Comment.source_post_id.label("source_post_id"),
                Post.post_id.label("post_id"),
                Post.platform.label("platform"),
                Campaign.campaign_id.label("campaign_id"),
                Campaign.campaign_name.label("campaign_name"),
                Product.product_id.label("product_id"),
                Product.product_name.label("product_name"),
                CommentAnalysis.topic.label("topic"),
                CommentAnalysis.sentiment.label("sentiment"),
                CommentAnalysis.side_effect_flag.label("side_effect_flag"),
                CommentAnalysis.reason.label("reason"),
                CommentAnalysis.analysed_hash.label("analysed_hash"),
                Comment.text_hash.label("text_hash"),
            )
            .select_from(Comment)
            .outerjoin(Post, Comment.post_id == Post.post_id)
            .outerjoin(Campaign, Post.campaign_id == Campaign.campaign_id)
            .outerjoin(Product, Campaign.product_id == Product.product_id)
            .outerjoin(CommentAnalysis, Comment.comment_id == CommentAnalysis.comment_id)
            .order_by(Comment.comment_id)
        ).mappings().all()
        reactions = db.execute(
            select(Product.product_name, Product.active_ingredient,
                   FDAReaction.reaction, FDAReaction.report_count)
            .select_from(FDAReaction)
            .join(Product, FDAReaction.product_id == Product.product_id)
            .order_by(Product.product_name, FDAReaction.report_count.desc())
        ).all()
    return pd.DataFrame(rows), pd.DataFrame(
        reactions, columns=["Product", "Active ingredient", "FDA reaction", "Reports"]
    )

try:
    comments, fda = fetch_data()
except Exception as e:
    st.error("Could not read SQLite data. First run your delivery pipeline and check that the database tables exist.")
    st.exception(e)
    st.stop()

if st.button("↻ Refresh database"):
    fetch_data.clear()
    st.rerun()

if comments.empty:
    st.info("There are no comments in SQLite yet. Run: python main.py data/delivery_1")
    st.stop()

# A classification is current only when it matches the comment's current text.
comments["classified"] = (
    comments["topic"].notna()
    & comments["analysed_hash"].notna()
    & (comments["analysed_hash"] == comments["text_hash"])
)
comments["flagged"] = comments["classified"] & comments["side_effect_flag"].eq(True)

with st.sidebar:
    st.header("Filters")
    products = sorted(comments["product_name"].dropna().unique().tolist())
    product_filter = st.multiselect("Product", products, default=products)
    campaigns = sorted(comments["campaign_name"].dropna().unique().tolist())
    campaign_filter = st.multiselect("Campaign", campaigns, default=campaigns)
    st.caption("Unlinked comments are included in the overview and the human-review table.")

linked = comments[
    comments["product_name"].isin(product_filter)
    & comments["campaign_name"].isin(campaign_filter)
].copy()
# Keep orphaned comments visible in the review queue, rather than losing safety signals.
review_source = comments[comments["flagged"]].copy()

all_tab, sentiment_tab, effects_tab, review_tab = st.tabs(
    ["Overview", "Campaign sentiment", "Side effects + openFDA", "Needs human review"]
)

with all_tab:
    a, b, c, d = st.columns(4)
    a.metric("Total comments", len(comments))
    b.metric("Classified (current)", int(comments["classified"].sum()))
    c.metric("Pending / changed", int((~comments["classified"]).sum()))
    d.metric("Possible side effects", int(comments["flagged"].sum()))
    st.info("Only successfully classified comments are included in sentiment and side-effect charts. Review the full dataset after classification is complete.")
    st.subheader("Classification breakdown")
    current = linked[linked["classified"]]
    if current.empty:
        st.warning("No classified comments for the current selection.")
    else:
        st.bar_chart(current["topic"].value_counts().rename_axis("Topic").rename("Count"))
        st.dataframe(
            current[["comment_id", "product_name", "campaign_name", "comment_text", "topic", "sentiment", "reason"]],
            use_container_width=True, hide_index=True,
        )

with sentiment_tab:
    st.subheader("Sentiment by campaign")
    current = linked[linked["classified"] & linked["campaign_name"].notna()]
    if current.empty:
        st.info("No classified comments linked to a campaign yet.")
    else:
        counts = (current.groupby(["campaign_name", "sentiment"]).size()
                  .unstack(fill_value=0).reindex(columns=["positive", "neutral", "negative"], fill_value=0))
        st.bar_chart(counts)
        st.dataframe(counts.reset_index(), use_container_width=True, hide_index=True)

with effects_tab:
    st.subheader("Possible side-effect comments by product")
    flagged = linked[linked["flagged"] & linked["product_name"].notna()]
    if flagged.empty:
        st.info("No flagged comments linked to products in the selected data.")
    else:
        counts = flagged.groupby("product_name").size().rename("Flagged comments")
        st.bar_chart(counts)
        st.dataframe(counts.reset_index(), use_container_width=True, hide_index=True)
    st.subheader("Reported reactions from openFDA")
    st.caption("openFDA counts are spontaneous adverse-event reports, not incidence rates or evidence of causation.")
    if fda.empty:
        st.warning("No openFDA reactions saved yet. Implement/run src/fda_client.py to populate fda_reactions.")
    else:
        selected = fda[fda["Product"].isin(product_filter)]
        st.dataframe(selected, use_container_width=True, hide_index=True)

with review_tab:
    st.subheader("Needs human review")
    st.warning("Possible side effects are screening flags only. Human review is required; unclassified comments have not yet been screened.")
    if review_source.empty:
        st.info("No comments have been flagged so far. This does not mean the full dataset is free of possible side effects.")
    else:
        cols = ["comment_id", "comment_text", "reason", "source_post_id", "post_id",
                "campaign_name", "product_name", "topic", "sentiment"]
        st.dataframe(review_source[cols], use_container_width=True, hide_index=True)
        st.download_button("Export flagged comments CSV",
                           review_source[cols].to_csv(index=False).encode("utf-8-sig"),
                           file_name="needs_human_review.csv", mime="text/csv")
