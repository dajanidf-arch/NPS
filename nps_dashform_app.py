import io
import re
import string
import base64
import zipfile
from collections import Counter

import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="LUGPA NPS Dashform",
    page_icon="📊",
    layout="wide",
)

st.title("📊 LUGPA NPS Dashform — Provider Experience Insights")
st.caption("Upload your NPS export to get provider-level NPS, theme analysis, and action suggestions.")

# -----------------------------
# Helper functions
# -----------------------------
REQUIRED_COLS = [
    "Practitioner Name",
    "NetPromoterScore",
    "Comments",
]

def classify_nps(score: float) -> str:
    try:
        score = float(score)
    except:
        return "unknown"
    if score >= 9:
        return "promoter"
    elif score >= 7:
        return "passive"
    elif score >= 0:
        return "detractor"
    return "unknown"

def compute_nps(series) -> float:
    # NPS = %Promoters - %Detractors
    n = len(series)
    if n == 0:
        return np.nan
    promoters = (series >= 9).sum() / n
    detractors = (series <= 6).sum() / n
    return round((promoters - detractors) * 100, 1)

# Theme taxonomy
THEME_TAXONOMY = {
    "wait_time": ["wait", "waiting", "delayed", "late"],
    "rushed": ["rushed", "in and out", "quickly"],
    "listening": ["didn't listen", "ignored", "unresponsive"],
    "communication": ["explain", "confusing", "understand", "questions"],
    "bedside_manner": ["rude", "kind", "caring", "compassion", "dismissive"],
    "staff": ["front desk", "reception", "staff", "nurse", "billing", "insurance"],
    "follow_up": ["follow up", "call back", "results", "refill", "response time"],
    "outcomes": ["pain", "complication", "improved", "worse", "side effect", "treatment"],
}

ACTION_PLAYBOOK = {
    "wait_time": [
        "Audit template lengths by visit type.",
        "Create buffer blocks in daily schedules.",
        "Send pre-visit SMS updates if running behind.",
    ],
    "communication": [
        "Use 'teach-back' to confirm patient understanding.",
        "Close visits with clear summary and next steps.",
        "Standardize MyChart reply macros for common questions.",
    ],
    "bedside_manner": [
        "Adopt a 90-second warm start (eye contact, open question).",
        "Mirror emotions once per visit; thank the patient for sharing.",
        "Role-play difficult conversations with a coach.",
    ],
    "staff": [
        "Weekly huddle to review patient friction points.",
        "Shadow front desk periodically and fix bottlenecks.",
        "Create a 'hot handoff' script from staff to clinicians.",
    ],
    "access": [
        "Offer quick telehealth slots for clarifications.",
        "Add voicemail-to-text routing for faster triage.",
        "Publish clear refill policy handouts.",
    ],
    "outcomes": [
        "Set expectations explicitly (risks, timelines, red flags).",
        "Provide a printed/portal summary of treatment plans.",
        "Schedule proactive check-ins for high-risk patients.",
    ],
}

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def map_themes(text: str):
    text_c = clean_text(text)
    hits = []
    for theme, kws in THEME_TAXONOMY.items():
        for kw in kws:
            if kw in text_c:
                hits.append(theme)
                break
    return hits

def top_phrases(texts, n=15):
    tokens = []
    for t in texts:
        if not isinstance(t, str):
            continue
        t = clean_text(t)
        for ch in string.punctuation:
            t = t.replace(ch, " ")
        words = [w for w in t.split() if len(w) > 2 and not w.isnumeric()]
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        tokens.extend(bigrams + trigrams)
    cnt = Counter(tokens)
    return cnt.most_common(n)

def df_to_download_link(df, filename="export.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# -----------------------------
# File uploader
# -----------------------------
st.subheader("1) Upload NPS data (CSV or Excel)")
uploaded = st.file_uploader("Select your NPS file", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.success(f"Loaded {len(df):,} rows from {uploaded.name}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["nps_bucket"] = df["NetPromoterScore"].apply(classify_nps)
    df["themes"] = df["Comments"].apply(map_themes)

    # -----------------------------
    # 2) Provider Summary
    # -----------------------------
    st.subheader("2) Provider summary")
    prov_group = df.groupby("Practitioner Name", dropna=False)
    summary = prov_group.agg(
        encounters=("NetPromoterScore", "size"),
        avg_score=("NetPromoterScore", "mean"),
        nps=("NetPromoterScore", compute_nps),
        detractors=("nps_bucket", lambda s: (s == "detractor").sum()),
        passives=("nps_bucket", lambda s: (s == "passive").sum()),
        promoters=("nps_bucket", lambda s: (s == "promoter").sum()),
    ).reset_index()
    summary["avg_score"] = summary["avg_score"].round(2)
    st.dataframe(summary.sort_values("nps", ascending=True), use_container_width=True)
    st.markdown(df_to_download_link(summary, "provider_summary.csv"), unsafe_allow_html=True)

    # -----------------------------
    # 3) Global Themes
    # -----------------------------
    st.subheader("3) Global themes")
    df_exploded = df.explode("themes")
    theme_counts = (
        df_exploded.dropna(subset=["themes"])
        .groupby("themes")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Theme frequency across all reviews")
        st.bar_chart(theme_counts.set_index("themes"))
    with col2:
        st.caption("Top phrases (bigrams/trigrams)")
        phrases = top_phrases(df["Comments"].tolist(), n=20)
        st.dataframe(pd.DataFrame(phrases, columns=["phrase", "count"]), use_container_width=True)

    # -----------------------------
    # 4) Provider Drilldown
    # -----------------------------
    st.subheader("4) Provider drilldown")
    provider_sel = st.selectbox(
        "Choose a provider",
        options=summary["Practitioner Name"].tolist()
    )
    dprov = df[df["Practitioner Name"] == provider_sel].copy()
    st.markdown(
        f"**Encounters:** {len(dprov):,} | "
        f"**Avg score:** {dprov['NetPromoterScore'].mean():.2f} | "
        f"**NPS:** {compute_nps(dprov['NetPromoterScore']):.1f}"
    )

    dprov_exploded = dprov.explode("themes")
    prov_theme_counts = (
        dprov_exploded.dropna(subset=["themes"])
        .groupby("themes")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if len(prov_theme_counts):
        st.bar_chart(prov_theme_counts.set_index("themes"))
    else:
        st.info("No detected themes for this provider.")

    with st.expander("See raw reviews"):
        show_cols = ["VisitDate", "NetPromoterScore", "nps_bucket", "Comments", "Place of Service", "Region"]
        show_cols = [c for c in show_cols if c in dprov.columns]
        st.dataframe(dprov[show_cols].sort_values("VisitDate", ascending=False), use_container_width=True)

    # -----------------------------
    # 5) Action Suggestions
    # -----------------------------
    st.subheader("5) Action suggestions")
    top_theme_list = prov_theme_counts["themes"].head(3).tolist() if len(prov_theme_counts) else []
    actions = []
    for t in top_theme_list:
        actions.extend(ACTION_PLAYBOOK.get(t, []))
    if actions:
        st.markdown("**Focus areas:** " + ", ".join(top_theme_list))
        for a in actions:
            st.markdown(f"- {a}")
    else:
        st.write("—")

    # -----------------------------
    # 6) Exports
    # -----------------------------
    st.subheader("6) Exports")
    prov_export = dprov.copy()
    prov_export_path = f"{provider_sel.replace(' ', '_').lower()}_reviews.csv"
    st.markdown(df_to_download_link(prov_export, prov_export_path), unsafe_allow_html=True)

else:
    st.info("Upload an Excel/CSV file with at least: Practitioner Name, NetPromoterScore, Comments")

