"""
CampusPulse AI — Smart Student Feedback Analyzer
Modern Dark-Themed Streamlit Application for Campus Feedback Sentiment & Category Intelligence.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Theme Settings
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CampusPulse AI — Student Feedback Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Modern Academic Dark Navy & Indigo/Cyan Accents
CUSTOM_CSS = """
<style>
    /* Global Base */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Modern Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #6366f1;
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .kpi-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }
    
    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: capitalize;
    }
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-neutral {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-cat {
        background-color: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Result Container Card */
    .result-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Dataframe table borders */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# File Paths & Model Loading
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RAW_DATA_PATH = DATA_DIR / "student_feedback.csv"
CLEAN_DATA_PATH = DATA_DIR / "student_feedback_cleaned.csv"
EVAL_RESULTS_PATH = REPORTS_DIR / "evaluation_results.json"
METADATA_PATH = MODELS_DIR / "metadata.json"


@st.cache_data
def load_dataset():
    """Loads student feedback dataset from disk."""
    path_to_load = CLEAN_DATA_PATH if CLEAN_DATA_PATH.exists() else RAW_DATA_PATH
    if not path_to_load.exists():
        return None
    df = pd.read_csv(path_to_load)
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip().str.lower().str.replace("_", " ")
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    return df


@st.cache_data
def load_evaluation_metrics():
    """Loads evaluation JSON metrics."""
    if not EVAL_RESULTS_PATH.exists():
        return None
    with open(EVAL_RESULTS_PATH, "r") as f:
        return json.load(f)


@st.cache_resource
def load_trained_pipelines():
    """Loads trained pipelines for sentiment and category classification."""
    models = {}
    
    # Best models
    sent_best_path = MODELS_DIR / "sentiment_best_model.joblib"
    cat_best_path = MODELS_DIR / "category_best_model.joblib"
    
    # Specific variants
    sent_nb_path = MODELS_DIR / "sentiment_nb_pipeline.joblib"
    sent_lr_path = MODELS_DIR / "sentiment_lr_pipeline.joblib"
    cat_nb_path = MODELS_DIR / "category_nb_pipeline.joblib"
    cat_lr_path = MODELS_DIR / "category_lr_pipeline.joblib"
    
    if sent_best_path.exists():
        models["sentiment_best"] = joblib.load(sent_best_path)
    if cat_best_path.exists():
        models["category_best"] = joblib.load(cat_best_path)
    if sent_nb_path.exists():
        models["sentiment_nb"] = joblib.load(sent_nb_path)
    if sent_lr_path.exists():
        models["sentiment_lr"] = joblib.load(sent_lr_path)
    if cat_nb_path.exists():
        models["category_nb"] = joblib.load(cat_nb_path)
    if cat_lr_path.exists():
        models["category_lr"] = joblib.load(cat_lr_path)
        
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            models["metadata"] = json.load(f)
    else:
        models["metadata"] = {}
        
    return models


# -----------------------------------------------------------------------------
# Color Palette Constants
# -----------------------------------------------------------------------------
SENTIMENT_COLORS = {
    "positive": "#10B981",  # Emerald
    "neutral": "#F59E0B",   # Amber
    "negative": "#EF4444",  # Rose
}
CATEGORY_COLORS = [
    "#6366F1", "#06B6D4", "#8B5CF6", "#EC4899", "#3B82F6", "#14B8A6", "#F97316"
]

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 CampusPulse AI")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>"
        "Smart Student Feedback Analyzer & Intelligence Dashboard"
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "🔍 Analyze Live Feedback",
            "📂 Feedback Explorer",
            "📈 Model Evaluation",
            "📖 About Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    
    df_data = load_dataset()
    models_dict = load_trained_pipelines()
    
    if df_data is not None:
        st.markdown(f"🟢 **Dataset:** Active ({len(df_data)} records)")
    else:
        st.markdown("🔴 **Dataset:** Not found (`data/student_feedback.csv`)")
        
    if "sentiment_best" in models_dict and "category_best" in models_dict:
        meta = models_dict.get("metadata", {})
        st.markdown("🟢 **Models:** Loaded in memory")
        st.caption(f"Sentiment: `{meta.get('best_sentiment_model', 'Trained')}`")
        st.caption(f"Category: `{meta.get('best_category_model', 'Trained')}`")
    else:
        st.markdown("🔴 **Models:** Not trained (`run train_model.py`)")

    st.markdown("---")
    st.caption("CampusPulse AI v1.0.0 | Academic Release")


# -----------------------------------------------------------------------------
# PAGE 1: EXECUTIVE DASHBOARD
# -----------------------------------------------------------------------------
if page == "📊 Executive Dashboard":
    st.title("📊 Student Feedback Analytics Dashboard")
    st.markdown(
        "Comprehensive real-time overview of institutional student sentiment, departmental categories, "
        "and critical areas needing attention."
    )
    
    if df_data is None:
        st.error(
            "⚠️ Dataset not found. Please place `student_feedback.csv` inside `data/` "
            "and run `python train_model.py`."
        )
        st.stop()
        
    # Top KPI Cards
    total_feedbacks = len(df_data)
    pos_count = (df_data["sentiment"] == "positive").sum()
    neu_count = (df_data["sentiment"] == "neutral").sum()
    neg_count = (df_data["sentiment"] == "negative").sum()
    categories_count = df_data["category"].nunique()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Records</div>
                <div class="kpi-value">{total_feedbacks:,}</div>
                <div class="kpi-sub">Student evaluations</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left: 4px solid #10b981;">
                <div class="kpi-title">Positive Feedback</div>
                <div class="kpi-value" style="color: #34d399;">{pos_count:,}</div>
                <div class="kpi-sub">{(pos_count/total_feedbacks*100):.1f}% satisfaction</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
                <div class="kpi-title">Neutral Feedback</div>
                <div class="kpi-value" style="color: #fbbf24;">{neu_count:,}</div>
                <div class="kpi-sub">{(neu_count/total_feedbacks*100):.1f}% neutral / mixed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left: 4px solid #ef4444;">
                <div class="kpi-title">Negative Feedback</div>
                <div class="kpi-value" style="color: #f87171;">{neg_count:,}</div>
                <div class="kpi-sub">{(neg_count/total_feedbacks*100):.1f}% urgent action</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col5:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left: 4px solid #6366f1;">
                <div class="kpi-title">Categories</div>
                <div class="kpi-value" style="color: #a5b4fc;">{categories_count}</div>
                <div class="kpi-sub">Academic & campus topics</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Analytics Row 1: Sentiment Donut & Category Bar
    c_left, c_right = st.columns([1, 1.2])
    
    with c_left:
        st.subheader("Sentiment Distribution")
        sent_counts = df_data["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        
        fig_donut = px.pie(
            sent_counts,
            values="count",
            names="sentiment",
            hole=0.55,
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#0f172a", width=2))
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c_right:
        st.subheader("Feedback Records by Category")
        cat_counts = df_data["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        
        fig_bar = px.bar(
            cat_counts,
            x="count",
            y="category",
            orientation="h",
            text="count",
            color="count",
            color_continuous_scale="Purples",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(showgrid=True, gridcolor="#334155", title="Number of Comments"),
            yaxis=dict(title="", categoryorder="total ascending"),
            coloraxis_showscale=False
        )
        fig_bar.update_traces(
            textposition="outside",
            marker=dict(line=dict(color="#6366f1", width=1))
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    
    # Visual Analytics Row 2: Category x Sentiment Breakdown & Negative Issues
    c_bot1, c_bot2 = st.columns([1.3, 1])
    
    with c_bot1:
        st.subheader("Sentiment Composition Across Campus Categories")
        ct_df = pd.crosstab(df_data["category"], df_data["sentiment"]).reset_index()
        
        fig_stacked = go.Figure()
        for sent in ["positive", "neutral", "negative"]:
            if sent in ct_df.columns:
                fig_stacked.add_trace(go.Bar(
                    name=sent.capitalize(),
                    y=ct_df["category"],
                    x=ct_df[sent],
                    orientation="h",
                    marker_color=SENTIMENT_COLORS.get(sent, "#94a3b8")
                ))
        fig_stacked.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(showgrid=True, gridcolor="#334155", title="Comments"),
            yaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
        
    with c_bot2:
        st.subheader("⚠️ Critical Attention Areas (Negative Feedback)")
        neg_df = df_data[df_data["sentiment"] == "negative"]
        if len(neg_df) > 0:
            neg_by_cat = neg_df["category"].value_counts().reset_index()
            neg_by_cat.columns = ["category", "negative_count"]
            
            fig_neg = px.bar(
                neg_by_cat,
                x="category",
                y="negative_count",
                text="negative_count",
                color="negative_count",
                color_continuous_scale="Reds",
            )
            fig_neg.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(title="Category", tickangle=-30),
                yaxis=dict(showgrid=True, gridcolor="#334155", title="Negative Comments"),
                coloraxis_showscale=False
            )
            fig_neg.update_traces(textposition="outside")
            st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info("No negative feedback recorded in the current dataset.")

    # Executive Takeaways Box
    st.markdown("### 📌 Strategic Insights & Key Takeaways")
    most_popular_cat = df_data["category"].mode()[0].title()
    top_negative_cat = (
        neg_df["category"].mode()[0].title() if len(neg_df) > 0 else "None"
    )
    
    st.markdown(
        f"""
        <div class="result-box">
            <ul style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.8; margin-bottom: 0;">
                <li><strong>Overall Satisfaction:</strong> <b>{(pos_count/total_feedbacks*100):.1f}%</b> of student submissions express positive reception towards university facilities and academics.</li>
                <li><strong>Highest Volume Area:</strong> The <b>{most_popular_cat}</b> category constitutes the largest volume of student commentary.</li>
                <li><strong>Primary Grievance Source:</strong> The <b>{top_negative_cat}</b> domain accounts for the highest density of negative reviews, highlighting priority for administrative interventions.</li>
                <li><strong>Data Nuance:</strong> Neutral comments (<b>{(neu_count/total_feedbacks*100):.1f}%</b>) frequently represent constructive feedback (e.g., duration of practicals, syllabus adjustments) rather than pure apathy.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# PAGE 2: ANALYZE LIVE FEEDBACK
# -----------------------------------------------------------------------------
elif page == "🔍 Analyze Live Feedback":
    st.title("🔍 Live Student Feedback Analyzer")
    st.markdown(
        "Enter student feedback below to test real-time classification through our trained NLP TF-IDF models. "
        "The system evaluates emotional valence and maps the comment to the most relevant campus category."
    )
    
    if "sentiment_best" not in models_dict or "category_best" not in models_dict:
        st.error(
            "⚠️ Trained models are missing. Please execute `python train_model.py` in your terminal first."
        )
        st.stop()
        
    sent_model = models_dict["sentiment_best"]
    cat_model = models_dict["category_best"]
    metadata = models_dict.get("metadata", {})
    
    # Preset sample selector for quick testing
    sample_options = [
        "-- Select a pre-loaded sample or write custom text below --",
        "The professor explains core theoretical concepts clearly and always clears doubts after class.",
        "The lab equipment is often faulty and the two hour sessions are not enough for experiments.",
        "Examination hall seating was congested and question pattern was quite difficult.",
        "Library hours should be extended during end semester examinations and Wi-Fi improved.",
        "The annual sports fest and robotics club activities were organized brilliantly.",
        "Lecture delivery was average, nothing special but attendance is compulsory.",
    ]
    
    chosen_sample = st.selectbox("Quick-fill with representative student comment:", sample_options)
    default_text = "" if chosen_sample.startswith("--") else chosen_sample
    
    feedback_text = st.text_area(
        "Student Feedback Input Text",
        value=default_text,
        placeholder="Type or paste student feedback here (e.g., 'The laboratory air conditioning has been broken for two weeks')...",
        height=130
    )
    
    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Feedback", type="primary", use_container_width=True)
        
    if analyze_clicked or feedback_text.strip():
        if not feedback_text.strip():
            st.warning("Please type a feedback comment before clicking Analyze.")
        else:
            # Prediction Inference
            pred_sentiment = sent_model.predict([feedback_text])[0]
            pred_category = cat_model.predict([feedback_text])[0]
            
            # Probability / Decision scores
            sent_conf = None
            if hasattr(sent_model, "predict_proba"):
                probs = sent_model.predict_proba([feedback_text])[0]
                sent_classes = sent_model.classes_
                sent_conf = dict(zip(sent_classes, probs))
                
            cat_conf = None
            if hasattr(cat_model, "predict_proba"):
                cat_probs = cat_model.predict_proba([feedback_text])[0]
                cat_classes = cat_model.classes_
                cat_conf = dict(zip(cat_classes, cat_probs))
                
            st.markdown("### Prediction Results")
            res1, res2 = st.columns(2)
            
            # Sentiment Card
            with res1:
                badge_class = f"badge-{pred_sentiment}"
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="kpi-title">Predicted Sentiment</div>
                        <div style="margin: 0.5rem 0;">
                            <span class="badge-pill {badge_class}" style="font-size: 1.25rem; padding: 0.4rem 1rem;">
                                {pred_sentiment.upper()}
                            </span>
                        </div>
                        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
                            Model: <code>{metadata.get('best_sentiment_model', 'TF-IDF Pipeline')}</code>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if sent_conf:
                    st.caption("Class Probability Distribution:")
                    for cls_name, prob in sent_conf.items():
                        c_bar = SENTIMENT_COLORS.get(cls_name, "#6366f1")
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; font-size:0.85rem;'>"
                            f"<span>{cls_name.capitalize()}</span><span><b>{prob*100:.1f}%</b></span></div>",
                            unsafe_allow_html=True
                        )
                        st.progress(float(prob))

            # Category Card
            with res2:
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="kpi-title">Predicted Category / Topic</div>
                        <div style="margin: 0.5rem 0;">
                            <span class="badge-pill badge-cat" style="font-size: 1.25rem; padding: 0.4rem 1rem;">
                                🏷️ {pred_category.title()}
                            </span>
                        </div>
                        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
                            Model: <code>{metadata.get('best_category_model', 'TF-IDF Pipeline')}</code>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if cat_conf:
                    st.caption("Top Category Probabilities:")
                    sorted_cats = sorted(cat_conf.items(), key=lambda x: x[1], reverse=True)[:4]
                    for cat_name, prob in sorted_cats:
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; font-size:0.85rem;'>"
                            f"<span>{cat_name.title()}</span><span><b>{prob*100:.1f}%</b></span></div>",
                            unsafe_allow_html=True
                        )
                        st.progress(float(prob))
                        
            # Model Limitations Notice
            st.markdown(
                """
                <div style="background-color: #1e1b4b; border: 1px solid #4338ca; border-radius: 10px; padding: 1rem; margin-top: 1.5rem;">
                    <h5 style="color: #a5b4fc; margin-bottom: 0.3rem;">ℹ️ Prediction Limitations & Academic Notes</h5>
                    <p style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 0;">
                        TF-IDF uses n-gram bag-of-words token representations. While effective for campus-specific vocabulary,
                        it does not model deep syntactic context or complex sarcasm. Extremely short ambiguous comments
                        (e.g., "fine") or multi-topic paragraphs may have lower classification confidence.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# -----------------------------------------------------------------------------
# PAGE 3: FEEDBACK EXPLORER
# -----------------------------------------------------------------------------
elif page == "📂 Feedback Explorer":
    st.title("📂 Student Feedback Explorer")
    st.markdown("Search, filter, inspect, and export student feedback records from the verified dataset.")
    
    if df_data is None:
        st.error("⚠️ Dataset not found. Please place `student_feedback.csv` in `data/`.")
        st.stop()
        
    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])
    with f_col1:
        search_query = st.text_input("🔍 Search comment text:", placeholder="Search keywords (e.g. practical, exam, teacher)...")
    with f_col2:
        all_sentiments = ["All"] + sorted(df_data["sentiment"].dropna().unique().tolist())
        selected_sentiment = st.selectbox("Filter by Sentiment:", all_sentiments)
    with f_col3:
        all_categories = ["All"] + sorted(df_data["category"].dropna().unique().tolist())
        selected_category = st.selectbox("Filter by Category:", all_categories)
        
    filtered_df = df_data.copy()
    if search_query.strip():
        filtered_df = filtered_df[filtered_df["text"].astype(str).str.contains(search_query.strip(), case=False, na=False)]
    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
        
    st.markdown(f"**Showing {len(filtered_df):,} of {len(df_data):,} records**")
    
    # Styled Display Table
    cols_to_show = [c for c in ["text", "category", "sentiment", "sentiment_score"] if c in filtered_df.columns]
    st.dataframe(
        filtered_df[cols_to_show],
        use_container_width=True,
        height=450,
        column_config={
            "text": st.column_config.TextColumn("Feedback Text", width="large"),
            "category": st.column_config.TextColumn("Category", width="medium"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "sentiment_score": st.column_config.NumberColumn("Score", width="small"),
        }
    )
    
    # Export Button
    csv_bytes = filtered_df[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Records as CSV",
        data=csv_bytes,
        file_name="campuspulse_filtered_feedback.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# PAGE 4: MODEL EVALUATION
# -----------------------------------------------------------------------------
elif page == "📈 Model Evaluation":
    st.title("📈 Machine Learning Model Evaluation & Benchmark")
    st.markdown(
        "Direct performance comparison between **Multinomial Naive Bayes** and **Logistic Regression** "
        "trained on unigram/bigram TF-IDF representations with stratified 80/20 train-test splits."
    )
    
    eval_metrics = load_evaluation_metrics()
    if eval_metrics is None:
        st.warning(
            "⚠️ Evaluation metrics file (`reports/evaluation_results.json`) not found. "
            "Please run `python train_model.py` to train and evaluate the models."
        )
        st.stop()
        
    # Tab selection for Sentiment vs Category
    tab_sentiment, tab_category = st.tabs(["🎯 Sentiment Classification Task", "🏷️ Category Classification Task"])
    
    with tab_sentiment:
        st.subheader("Task 1: Sentiment Classification Performance")
        sent_info = eval_metrics.get("sentiment_models", {})
        nb_sent = sent_info.get("Naive Bayes", {})
        lr_sent = sent_info.get("Logistic Regression", {})
        
        # Metric comparison table
        comp_rows = [
            {
                "Metric": "Overall Accuracy",
                "Naive Bayes": f"{nb_sent.get('accuracy', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_sent.get('accuracy', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_sent.get('accuracy', 0) - nb_sent.get('accuracy', 0))*100:+.2f}%"
            },
            {
                "Metric": "Macro Precision",
                "Naive Bayes": f"{nb_sent.get('precision_macro', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_sent.get('precision_macro', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_sent.get('precision_macro', 0) - nb_sent.get('precision_macro', 0))*100:+.2f}%"
            },
            {
                "Metric": "Macro Recall",
                "Naive Bayes": f"{nb_sent.get('recall_macro', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_sent.get('recall_macro', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_sent.get('recall_macro', 0) - nb_sent.get('recall_macro', 0))*100:+.2f}%"
            },
            {
                "Metric": "Macro F1-Score (Primary)",
                "Naive Bayes": f"{nb_sent.get('f1_macro', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_sent.get('f1_macro', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_sent.get('f1_macro', 0) - nb_sent.get('f1_macro', 0))*100:+.2f}%"
            },
            {
                "Metric": "Weighted F1-Score",
                "Naive Bayes": f"{nb_sent.get('f1_weighted', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_sent.get('f1_weighted', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_sent.get('f1_weighted', 0) - nb_sent.get('f1_weighted', 0))*100:+.2f}%"
            },
        ]
        st.table(pd.DataFrame(comp_rows))
        
        # Visual Figures
        col_f1, col_f2 = st.columns(2)
        sent_cm_fig = FIGURES_DIR / "sentiment_confusion_matrix.png"
        sent_comp_fig = FIGURES_DIR / "sentiment_model_comparison.png"
        
        with col_f1:
            st.markdown("#### Confusion Matrix (Selected Model)")
            if sent_cm_fig.exists():
                st.image(str(sent_cm_fig), use_container_width=True)
            else:
                st.info("Confusion matrix image pending generation.")
                
        with col_f2:
            st.markdown("#### Metric Benchmark Bar Chart")
            if sent_comp_fig.exists():
                st.image(str(sent_comp_fig), use_container_width=True)
            else:
                st.info("Comparison chart pending generation.")
                
        # Per-class Classification Report
        st.markdown("#### Detailed Classification Report (Selected Model)")
        best_name = sent_info.get("selected_best", "Logistic Regression")
        target_metrics = lr_sent if best_name == "Logistic Regression" else nb_sent
        clf_rep = target_metrics.get("classification_report", {})
        
        rep_rows = []
        for label in sent_info.get("classes", []):
            if label in clf_rep:
                rep_rows.append({
                    "Class": label.capitalize(),
                    "Precision": f"{clf_rep[label]['precision']*100:.1f}%",
                    "Recall": f"{clf_rep[label]['recall']*100:.1f}%",
                    "F1-Score": f"{clf_rep[label]['f1-score']*100:.1f}%",
                    "Support": int(clf_rep[label]['support'])
                })
        if rep_rows:
            st.dataframe(pd.DataFrame(rep_rows), use_container_width=True)

    with tab_category:
        st.subheader("Task 2: Category Classification Performance")
        cat_info = eval_metrics.get("category_models", {})
        nb_cat = cat_info.get("Naive Bayes", {})
        lr_cat = cat_info.get("Logistic Regression", {})
        
        comp_cat_rows = [
            {
                "Metric": "Overall Accuracy",
                "Naive Bayes": f"{nb_cat.get('accuracy', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_cat.get('accuracy', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_cat.get('accuracy', 0) - nb_cat.get('accuracy', 0))*100:+.2f}%"
            },
            {
                "Metric": "Macro Precision",
                "Naive Bayes": f"{nb_cat.get('precision_macro', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_cat.get('precision_macro', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_cat.get('precision_macro', 0) - nb_cat.get('precision_macro', 0))*100:+.2f}%"
            },
            {
                "Metric": "Macro Recall",
                "Naive Bayes": f"{nb_cat.get('recall_macro', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_cat.get('recall_macro', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_cat.get('recall_macro', 0) - nb_cat.get('recall_macro', 0))*100:+.2f}%"
            },
            {
                "Metric": "Macro F1-Score (Primary)",
                "Naive Bayes": f"{nb_cat.get('f1_macro', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_cat.get('f1_macro', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_cat.get('f1_macro', 0) - nb_cat.get('f1_macro', 0))*100:+.2f}%"
            },
            {
                "Metric": "Weighted F1-Score",
                "Naive Bayes": f"{nb_cat.get('f1_weighted', 0)*100:.2f}%",
                "Logistic Regression": f"{lr_cat.get('f1_weighted', 0)*100:.2f}%",
                "Delta (LR - NB)": f"{(lr_cat.get('f1_weighted', 0) - nb_cat.get('f1_weighted', 0))*100:+.2f}%"
            },
        ]
        st.table(pd.DataFrame(comp_cat_rows))
        
        col_c1, col_c2 = st.columns(2)
        cat_cm_fig = FIGURES_DIR / "category_confusion_matrix.png"
        cat_comp_fig = FIGURES_DIR / "category_model_comparison.png"
        
        with col_c1:
            st.markdown("#### Confusion Matrix (Selected Model)")
            if cat_cm_fig.exists():
                st.image(str(cat_cm_fig), use_container_width=True)
            else:
                st.info("Category confusion matrix pending generation.")
                
        with col_c2:
            st.markdown("#### Metric Benchmark Bar Chart")
            if cat_comp_fig.exists():
                st.image(str(cat_comp_fig), use_container_width=True)
            else:
                st.info("Category benchmark chart pending generation.")

    # Model Performance & Selection Discussion
    st.markdown("---")
    st.markdown("### 🔬 Comparative Analysis & Model Selection Rationale")
    st.markdown(
        """
        <div class="result-box">
            <h5 style="color: #f1f5f9; margin-bottom: 0.5rem;">Why Model Selection Matters for Class Imbalance</h5>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                1. <strong>Multinomial Naive Bayes (MNB):</strong> Computes conditional word probability distributions under a conditional independence assumption. 
                While highly computationally efficient with small corpora, MNB often suffers from frequency bias towards the majority class (positive feedback), yielding lower recall on negative and neutral categories.<br>
                2. <strong>Logistic Regression (with Class Weighting):</strong> Employs convex loss optimization with L2 regularization and inverse class frequency weighting (<code>class_weight='balanced'</code>).
                This directly counters sample imbalance by penalizing classification errors on under-represented classes proportionally, resulting in superior Macro-F1 scores across both sentiment and category classification tasks.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# PAGE 5: ABOUT PROJECT
# -----------------------------------------------------------------------------
elif page == "📖 About Project":
    st.title("📖 About CampusPulse AI")
    st.markdown("### Smart Student Feedback Analysis System — Technical Documentation")
    
    st.markdown(
        """
        <div class="result-box">
            <h4>1. Problem Statement</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                Higher education institutions collect substantial volumes of qualitative student evaluations across semesters. 
                Manual appraisal is subjective, time-intensive, and delays timely pedagogical interventions. 
                CampusPulse AI automates qualitative feedback mining, extracting actionable sentiment polarity and topical category classifications.
            </p>
            
            <h4>2. Project Objectives</h4>
            <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                <li>Deliver an end-to-end NLP pipeline for dual classification: 3-class sentiment (positive, neutral, negative) and multi-class campus department categorization.</li>
                <li>Compare supervised algorithms (Multinomial Naive Bayes vs. Logistic Regression) using rigorous Macro-F1 and confusion matrix evaluation.</li>
                <li>Provide college leadership with an interactive, modern analytics dashboard to identify friction points and student satisfaction trends.</li>
            </ul>

            <h4>3. Dataset Source & Attribution</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                Derived from the Hugging Face dataset <code>Hannes111/student_feedback_sentiment</code> (based on the Kaggle Student Feedback dataset by <i>brarajit18</i>). 
                Contains verified student comments annotated with category tags and sentiment scores.
            </p>

            <h4>4. NLP Pipeline & Methodology</h4>
            <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                <li><strong>Text Preprocessing:</strong> Lowercasing, removal of non-alphanumeric noise, regex whitespace normalization, and negation preservation.</li>
                <li><strong>Feature Engineering:</strong> Term Frequency-Inverse Document Frequency (TF-IDF) utilizing unigram and bigram n-grams (1, 2), sublinear term frequency scaling, and vocabulary filtering to capture idiomatic phrasing (e.g., "not good", "very clear").</li>
                <li><strong>Leakage Prevention:</strong> Feature transformers are fitted exclusively on the training partition and transformed on test sets.</li>
                <li><strong>Model Training:</strong> Multinomial Naive Bayes (Laplace smoothing) alongside L2-penalized Logistic Regression with balanced class weights.</li>
            </ul>

            <h4>5. Academic Architecture Flow</h4>
            <pre style="background: #0f172a; padding: 1rem; border-radius: 8px; color: #38bdf8; font-family: monospace;">
[Raw Feedback CSV]
       │
       ▼
[Data Cleaning & Deduplication]
       │
       ▼
[Stratified 80/20 Train-Test Split]
       │
       ├──► [Training Partition] ──► [TF-IDF (1, 2) Fit] ──► [Model Fit (NB / LR)]
       │                                                          │
       └──► [Test Partition]     ──► [TF-IDF Transform]  ──► [Evaluation & Benchmark]
                                                                  │
                                                          [joblib Serialized Models]
                                                                  │
                                                                  ▼
                                                      [Streamlit Web Application]
            </pre>

            <h4>6. Ethical Considerations & Limitations</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                The dataset displays natural class imbalance with a predominance of positive remarks. 
                TF-IDF representation relies on surface n-grams and may miss subtle sarcasm or deeply idiomatic syntax. 
                Predictions are intended as decision-support signals rather than punitive measures.
            </p>

            <h4>7. Key References</h4>
            <ol style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                <li>Jurafsky, D., & Martin, J. H. (2023). <em>Speech and Language Processing</em> (3rd ed. draft).</li>
                <li>Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. <em>JMLR</em>, 12, 2825-2830.</li>
                <li>Hugging Face Datasets: <code>Hannes111/student_feedback_sentiment</code>.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
