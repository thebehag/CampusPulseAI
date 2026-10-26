"""
CampusPulse AI — NLP & Machine Learning Training Pipeline
Author: CampusPulse AI Team
Description: Trains and compares Multinomial Naive Bayes and Logistic Regression
             classifiers for student feedback sentiment and category classification.
             Uses TF-IDF with unigrams + bigrams, stratified splitting, and prevents data leakage.
"""

import json
import os
import re
import string
import tempfile
from pathlib import Path

# Ensure matplotlib uses standard writable temp directory
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """Cleans and normalizes text for NLP feature extraction.
    
    Preserves sentiment and negation cues while removing noisy formatting.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Replace URLs and email addresses
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove HTML tags if any
    text = re.sub(r"<.*?>", " ", text)
    # Replace uncommon symbols while keeping alphanumeric, spaces, and essential apostrophes
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    # Collapse duplicate whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_classifier(pipeline, X_test, y_test, class_names):
    """Evaluates a trained classifier pipeline and returns standard metrics."""
    y_pred = pipeline.predict(X_test)
    
    acc = float(accuracy_score(y_test, y_pred))
    p_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred, labels=class_names).tolist()
    clf_report = classification_report(y_test, y_pred, labels=class_names, output_dict=True, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "precision_weighted": p_weighted,
        "recall_macro": r_macro,
        "recall_weighted": r_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "test_size": len(y_test)
    }


def plot_and_save_confusion_matrix(cm, labels, title, save_path):
    """Generates and saves a clean, publication-ready confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cm_array = np.array(cm)
    
    cax = ax.matshow(cm_array, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    
    # Loop over data dimensions and create text annotations
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            val = cm_array[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontweight="bold")
            
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_and_save_comparison_bar(results, title, save_path):
    """Saves a side-by-side metric comparison bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    
    nb_scores = [results["Naive Bayes"][m] for m in metrics]
    lr_scores = [results["Logistic Regression"][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, nb_scores, width, label="Multinomial Naive Bayes", color="#4F46E5")
    bars2 = ax.bar(x + width/2, lr_scores, width, label="Logistic Regression", color="#06B6D4")
    
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
                    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def train_and_evaluate():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "student_feedback.csv"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset missing at {data_path}.\n"
            "Please ensure data/student_feedback.csv exists before running train_model.py."
        )
        
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Initial raw record count: {len(df)}")
    
    # Drop rows missing text, sentiment, or category
    df = df.dropna(subset=["text", "sentiment", "category"]).copy()
    # Normalize category names (replace underscores with spaces, lowercase)
    df["category"] = df["category"].astype(str).str.strip().str.lower().str.replace("_", " ")
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Filter empty cleaned texts
    df = df[df["clean_text"].str.strip().str.len() > 0].copy()
    print(f"Usable cleaned record count: {len(df)}")
    
    # Save cleaned copy with standardized categories for app reference
    df.to_csv(data_path.parent / "student_feedback_cleaned.csv", index=False)
    
    RANDOM_SEED = 42
    
    # -------------------------------------------------------------
    # 1. TASK 1: SENTIMENT CLASSIFICATION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 1: SENTIMENT CLASSIFICATION (Positive / Neutral / Negative)")
    print("=" * 60)
    
    sentiment_labels = sorted(df["sentiment"].unique())
    print(f"Sentiment classes: {sentiment_labels}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["sentiment"]
    )
    print(f"Sentiment train size: {len(X_train_s)}, test size: {len(X_test_s)}")
    
    # Sentiment Naive Bayes
    sentiment_nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    sentiment_nb_pipeline.fit(X_train_s, y_train_s)
    sentiment_nb_metrics = evaluate_classifier(sentiment_nb_pipeline, X_test_s, y_test_s, sentiment_labels)
    
    # Sentiment Logistic Regression
    sentiment_lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=RANDOM_SEED))
    ])
    sentiment_lr_pipeline.fit(X_train_s, y_train_s)
    sentiment_lr_metrics = evaluate_classifier(sentiment_lr_pipeline, X_test_s, y_test_s, sentiment_labels)
    
    print("\n--- Sentiment Model Comparison ---")
    print(f"Naive Bayes        -> Acc: {sentiment_nb_metrics['accuracy']:.4f} | Macro-F1: {sentiment_nb_metrics['f1_macro']:.4f} | Weighted-F1: {sentiment_nb_metrics['f1_weighted']:.4f}")
    print(f"Logistic Regression -> Acc: {sentiment_lr_metrics['accuracy']:.4f} | Macro-F1: {sentiment_lr_metrics['f1_macro']:.4f} | Weighted-F1: {sentiment_lr_metrics['f1_weighted']:.4f}")
    
    # Select best sentiment model based on Macro-F1 (handles class imbalance)
    if sentiment_lr_metrics["f1_macro"] >= sentiment_nb_metrics["f1_macro"]:
        best_sentiment_name = "Logistic Regression"
        best_sentiment_pipeline = sentiment_lr_pipeline
    else:
        best_sentiment_name = "Naive Bayes"
        best_sentiment_pipeline = sentiment_nb_pipeline
    print(f">> Selected Best Sentiment Model: {best_sentiment_name}")
    
    # -------------------------------------------------------------
    # 2. TASK 2: CATEGORY / TOPIC CLASSIFICATION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 2: CATEGORY CLASSIFICATION")
    print("=" * 60)
    
    # Check rare classes before stratifying
    cat_counts = df["category"].value_counts()
    valid_categories = cat_counts[cat_counts >= 2].index.tolist()
    df_cat = df[df["category"].isin(valid_categories)].copy()
    
    category_labels = sorted(df_cat["category"].unique())
    print(f"Category classes ({len(category_labels)}): {category_labels}")
    print(f"Category distribution:\n{df_cat['category'].value_counts()}")
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        df_cat["clean_text"],
        df_cat["category"],
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df_cat["category"]
    )
    print(f"Category train size: {len(X_train_c)}, test size: {len(X_test_c)}")
    
    # Category Naive Bayes
    category_nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    category_nb_pipeline.fit(X_train_c, y_train_c)
    category_nb_metrics = evaluate_classifier(category_nb_pipeline, X_test_c, y_test_c, category_labels)
    
    # Category Logistic Regression
    category_lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=RANDOM_SEED))
    ])
    category_lr_pipeline.fit(X_train_c, y_train_c)
    category_lr_metrics = evaluate_classifier(category_lr_pipeline, X_test_c, y_test_c, category_labels)
    
    print("\n--- Category Model Comparison ---")
    print(f"Naive Bayes        -> Acc: {category_nb_metrics['accuracy']:.4f} | Macro-F1: {category_nb_metrics['f1_macro']:.4f} | Weighted-F1: {category_nb_metrics['f1_weighted']:.4f}")
    print(f"Logistic Regression -> Acc: {category_lr_metrics['accuracy']:.4f} | Macro-F1: {category_lr_metrics['f1_macro']:.4f} | Weighted-F1: {category_lr_metrics['f1_weighted']:.4f}")
    
    # Select best category model
    if category_lr_metrics["f1_macro"] >= category_nb_metrics["f1_macro"]:
        best_category_name = "Logistic Regression"
        best_category_pipeline = category_lr_pipeline
    else:
        best_category_name = "Naive Bayes"
        best_category_pipeline = category_nb_pipeline
    print(f">> Selected Best Category Model: {best_category_name}")
    
    # -------------------------------------------------------------
    # 3. SAVE TRAINED PIPELINES
    # -------------------------------------------------------------
    print("\nSaving trained models to 'models/' directory...")
    joblib.dump(best_sentiment_pipeline, models_dir / "sentiment_best_model.joblib")
    joblib.dump(sentiment_nb_pipeline, models_dir / "sentiment_nb_pipeline.joblib")
    joblib.dump(sentiment_lr_pipeline, models_dir / "sentiment_lr_pipeline.joblib")
    
    joblib.dump(best_category_pipeline, models_dir / "category_best_model.joblib")
    joblib.dump(category_nb_pipeline, models_dir / "category_nb_pipeline.joblib")
    joblib.dump(category_lr_pipeline, models_dir / "category_lr_pipeline.joblib")
    
    # Save label encoders / class definitions
    metadata = {
        "sentiment_classes": sentiment_labels,
        "category_classes": category_labels,
        "best_sentiment_model": best_sentiment_name,
        "best_category_model": best_category_name,
        "random_seed": RANDOM_SEED,
        "total_raw_records": len(df),
        "total_usable_records": len(df_cat)
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # -------------------------------------------------------------
    # 4. SAVE EVALUATION RESULTS & GENERATE FIGURES
    # -------------------------------------------------------------
    evaluation_summary = {
        "dataset_statistics": {
            "total_records": len(df),
            "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
            "category_distribution": df_cat["category"].value_counts().to_dict(),
            "test_split_ratio": 0.20
        },
        "sentiment_models": {
            "Naive Bayes": sentiment_nb_metrics,
            "Logistic Regression": sentiment_lr_metrics,
            "selected_best": best_sentiment_name,
            "classes": sentiment_labels
        },
        "category_models": {
            "Naive Bayes": category_nb_metrics,
            "Logistic Regression": category_lr_metrics,
            "selected_best": best_category_name,
            "classes": category_labels
        }
    }
    
    eval_json_path = reports_dir / "evaluation_results.json"
    with open(eval_json_path, "w") as f:
        json.dump(evaluation_summary, f, indent=4)
    print(f"Evaluation results saved to {eval_json_path}")
    
    # Generate and save figures
    print("Generating figures in reports/figures/...")
    
    # 1. Confusion Matrix - Sentiment (Best Model)
    best_sent_cm = sentiment_lr_metrics["confusion_matrix"] if best_sentiment_name == "Logistic Regression" else sentiment_nb_metrics["confusion_matrix"]
    plot_and_save_confusion_matrix(
        best_sent_cm,
        sentiment_labels,
        f"Sentiment Confusion Matrix ({best_sentiment_name})",
        figures_dir / "sentiment_confusion_matrix.png"
    )
    
    # 2. Confusion Matrix - Category (Best Model)
    best_cat_cm = category_lr_metrics["confusion_matrix"] if best_category_name == "Logistic Regression" else category_nb_metrics["confusion_matrix"]
    plot_and_save_confusion_matrix(
        best_cat_cm,
        category_labels,
        f"Category Confusion Matrix ({best_category_name})",
        figures_dir / "category_confusion_matrix.png"
    )
    
    # 3. Model Comparison Bar Chart - Sentiment
    plot_and_save_comparison_bar(
        evaluation_summary["sentiment_models"],
        "Sentiment Classification: Naive Bayes vs Logistic Regression",
        figures_dir / "sentiment_model_comparison.png"
    )
    
    # 4. Model Comparison Bar Chart - Category
    plot_and_save_comparison_bar(
        evaluation_summary["category_models"],
        "Category Classification: Naive Bayes vs Logistic Regression",
        figures_dir / "category_model_comparison.png"
    )
    
    print("[SUCCESS] All models trained, evaluated, figures generated, and artifacts saved!")
    return evaluation_summary


if __name__ == "__main__":
    train_and_evaluate()
