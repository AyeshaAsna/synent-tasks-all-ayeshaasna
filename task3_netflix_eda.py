import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = r"D:\synent_tech_internship"
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "task3_netflix", "netflix_titles.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "tasks", "task3")


def extract_duration_num(duration: str) -> float:
    if pd.isna(duration):
        return float("nan")
    token = str(duration).split(" ")[0]
    return pd.to_numeric(token, errors="coerce")


def main(show_plots: bool) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("[Task3] Loading Netflix dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"[Task3] Shape: {df.shape}")

    # Basic cleaning for EDA.
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["duration_num"] = df["duration"].apply(extract_duration_num)
    df["year_added"] = df["date_added"].dt.year

    # Summary statistics.
    summary = df.describe(include="all").transpose()
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
    print("[Task3] Summary statistics saved.")

    # Correlation analysis on numeric columns.
    numeric_df = df.select_dtypes(include=["number"]).copy()
    corr = numeric_df.corr(numeric_only=True)
    corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
    print("[Task3] Correlation matrix saved.")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Netflix Numeric Correlation Heatmap")
    plt.tight_layout()
    corr_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=150)
    print(f"[Task3] Saved: {corr_path}")
    if not show_plots:
        plt.close()

    # Trend identification: titles added by year.
    trend = df["year_added"].value_counts().sort_index()
    trend.to_csv(os.path.join(OUTPUT_DIR, "titles_added_by_year.csv"), header=["count"])
    print("[Task3] Trend file saved.")

    plt.figure(figsize=(10, 5))
    trend.plot(kind="line", marker="o")
    plt.title("Trend: Netflix Titles Added by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Titles Added")
    plt.tight_layout()
    trend_path = os.path.join(OUTPUT_DIR, "trend_titles_by_year.png")
    plt.savefig(trend_path, dpi=150)
    print(f"[Task3] Saved: {trend_path}")
    if not show_plots:
        plt.close()

    # Content type comparison chart.
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="type", hue="type", palette="Set2", legend=False)
    plt.title("Movies vs TV Shows")
    plt.tight_layout()
    type_path = os.path.join(OUTPUT_DIR, "type_distribution.png")
    plt.savefig(type_path, dpi=150)
    print(f"[Task3] Saved: {type_path}")
    if show_plots:
        print("[Task3] Displaying plots... close plot windows to finish script.")
        plt.show()
    else:
        plt.close("all")

    # Insights report.
    top_year = trend.idxmax() if not trend.empty else "N/A"
    top_year_count = int(trend.max()) if not trend.empty else 0
    with open(os.path.join(OUTPUT_DIR, "eda_insights.txt"), "w", encoding="utf-8") as f:
        f.write("Task 3: Netflix EDA Insights\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Dataset rows: {len(df)}\n")
        f.write(f"Dataset columns: {len(df.columns)}\n")
        f.write(f"Most titles were added in year: {top_year} ({top_year_count} titles)\n")
        f.write("Generated artifacts:\n")
        f.write("- summary_statistics.csv\n")
        f.write("- correlation_matrix.csv\n")
        f.write("- correlation_heatmap.png\n")
        f.write("- trend_titles_by_year.png\n")
        f.write("- type_distribution.png\n")

    print(f"[Task3] Completed. EDA outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 Netflix EDA")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows; only save files.")
    args = parser.parse_args()
    main(show_plots=not args.no_show)
