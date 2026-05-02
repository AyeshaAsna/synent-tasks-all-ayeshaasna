import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = r"D:\synent_tech_internship"
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "Iris.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "tasks", "task2")


def main(show_plots: bool) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("[Task2] Loading Iris dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"[Task2] Shape: {df.shape}")

    # Bar chart: species count.
    print("[Task2] Creating bar chart...")
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="Species",
        order=df["Species"].value_counts().index,
        hue="Species",
        palette="viridis",
        legend=False,
    )
    plt.title("Iris Species Count")
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "bar_species_count.png")
    plt.savefig(bar_path, dpi=150)
    print(f"[Task2] Saved: {bar_path}")
    if not show_plots:
        plt.close()

    # Histogram: Sepal length distribution.
    print("[Task2] Creating histogram...")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["SepalLengthCm"], bins=20, kde=True, color="steelblue")
    plt.title("Histogram of Sepal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR, "hist_sepal_length.png")
    plt.savefig(hist_path, dpi=150)
    print(f"[Task2] Saved: {hist_path}")
    if not show_plots:
        plt.close()

    # Scatter plot: compare petal dimensions by species.
    print("[Task2] Creating scatter plot...")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species", palette="deep")
    plt.title("Petal Length vs Petal Width")
    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, "scatter_petal_length_width.png")
    plt.savefig(scatter_path, dpi=150)
    print(f"[Task2] Saved: {scatter_path}")
    if not show_plots:
        plt.close()

    # Feature comparison matrix.
    print("[Task2] Creating pairplot for feature comparison...")
    pair = sns.pairplot(
        df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]],
        hue="Species",
        corner=True,
    )
    pairplot_path = os.path.join(OUTPUT_DIR, "feature_comparison_pairplot.png")
    pair.savefig(pairplot_path, dpi=150)
    print(f"[Task2] Saved: {pairplot_path}")

    if show_plots:
        print("[Task2] Displaying plots... close plot windows to finish script.")
        plt.show()
    else:
        plt.close("all")

    print(f"[Task2] Completed. Visualizations saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 Iris visualizations")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot windows; only save output files.",
    )
    args = parser.parse_args()
    main(show_plots=not args.no_show)
