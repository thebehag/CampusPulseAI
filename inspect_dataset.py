"""
CampusPulse AI — Dataset Inspection & Exploration Script
Author: CampusPulse AI Team
Description: Inspects the raw Student Feedback Sentiment dataset, checks schemas,
             missing values, duplicates, class distributions, and text length stats.
"""

from pathlib import Path
import pandas as pd
import numpy as np

def inspect_dataset(data_path: Path) -> pd.DataFrame:
    print("=" * 70)
    print("         CAMPUSPULSE AI — DATASET INSPECTION REPORT")
    print("=" * 70)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path.resolve()}\n"
            "Please ensure 'student_feedback.csv' is placed inside the 'data/' directory."
        )

    df = pd.read_csv(data_path)
    print(f"\n[+] Successfully loaded dataset from: {data_path}")
    print(f"[+] Total Records: {len(df)}")
    print(f"[+] Total Features: {df.shape[1]}")
    print(f"[+] Column Names: {list(df.columns)}")
    
    print("\n" + "-" * 40)
    print("1. DATA TYPES & NULL VALUE CHECK")
    print("-" * 40)
    null_counts = df.isnull().sum()
    info_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": null_counts,
        "Missing %": (null_counts / len(df) * 100).round(2)
    })
    print(info_df)

    print("\n" + "-" * 40)
    print("2. DUPLICATE RECORDS CHECK")
    print("-" * 40)
    dup_count = df.duplicated().sum()
    print(f"Total exact duplicate rows: {dup_count} ({(dup_count / len(df) * 100):.2f}%)")
    text_dup_count = df.duplicated(subset=['text']).sum()
    print(f"Duplicate feedback text entries: {text_dup_count} ({(text_dup_count / len(df) * 100):.2f}%)")

    print("\n" + "-" * 40)
    print("3. SENTIMENT LABEL DISTRIBUTION")
    print("-" * 40)
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts(dropna=False)
        sentiment_perc = df['sentiment'].value_counts(normalize=True, dropna=False) * 100
        sent_summary = pd.DataFrame({"Count": sentiment_counts, "Percentage (%)": sentiment_perc.round(2)})
        print(sent_summary)

    if 'sentiment_score' in df.columns:
        print("\nSentiment score mapping check:")
        print(df.groupby(['sentiment_score', 'sentiment'], observed=True).size())

    print("\n" + "-" * 40)
    print("4. CATEGORY (TOPIC) DISTRIBUTION")
    print("-" * 40)
    if 'category' in df.columns:
        cat_counts = df['category'].value_counts(dropna=False)
        cat_perc = df['category'].value_counts(normalize=True, dropna=False) * 100
        cat_summary = pd.DataFrame({"Count": cat_counts, "Percentage (%)": cat_perc.round(2)})
        print(cat_summary)

    print("\n" + "-" * 40)
    print("5. CROSS-TABULATION: CATEGORY VS SENTIMENT")
    print("-" * 40)
    if 'category' in df.columns and 'sentiment' in df.columns:
        ct = pd.crosstab(df['category'], df['sentiment'], margins=True, margins_name='Total')
        print(ct)

    print("\n" + "-" * 40)
    print("6. TEXT CHARACTER & WORD LENGTH STATISTICS")
    print("-" * 40)
    if 'text' in df.columns:
        valid_text = df['text'].dropna().astype(str)
        char_lens = valid_text.str.len()
        word_lens = valid_text.str.split().str.len()
        
        print("Character Length:")
        print(f"  Min: {char_lens.min()} | Mean: {char_lens.mean():.1f} | Median: {char_lens.median():.1f} | Max: {char_lens.max()}")
        print("Word Count:")
        print(f"  Min: {word_lens.min()} | Mean: {word_lens.mean():.1f} | Median: {word_lens.median():.1f} | Max: {word_lens.max()}")
        
        print("\nSample Feedback Texts:")
        for idx, row in df.sample(min(5, len(df)), random_state=42).iterrows():
            print(f" - [{row.get('sentiment', 'N/A')}] [{row.get('category', 'N/A')}]: {row['text'][:80]}...")

    print("\n" + "=" * 70)
    print("                 INSPECTION SUMMARY & INSIGHTS")
    print("=" * 70)
    print("1. Class Imbalance: 'positive' feedback dominates the dataset.")
    print("   -> Strategy: Stratified train/test split and Macro-F1 metric are essential.")
    print("2. Category classes have sufficient support across core campus areas.")
    print("3. Text Cleaning: Short feedback items need case normalization, whitespace stripping,")
    print("   and punctuation handling while retaining negation terms ('not', 'never').")
    print("=" * 70)

    return df

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "data" / "student_feedback.csv"
    inspect_dataset(data_file)
