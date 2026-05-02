import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = r"D:\synent_tech_internship"
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "Mall_Customers.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "tasks", "task6")


def main(show_plots: bool) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("[Task6] Loading mall customer dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"[Task6] Shape: {df.shape}")
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features].copy()

    # Preprocessing.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[Task6] Preprocessing done.")

    # K-Means clustering.
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    print("[Task6] K-Means clustering completed.")
    df.to_csv(os.path.join(OUTPUT_DIR, "mall_customers_with_clusters.csv"), index=False)

    # Cluster visualization.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="tab10",
        s=70,
    )
    plt.title("Customer Segments (K-Means)")
    plt.tight_layout()
    cluster_path = os.path.join(OUTPUT_DIR, "customer_clusters.png")
    plt.savefig(cluster_path, dpi=150)
    print(f"[Task6] Saved: {cluster_path}")
    if show_plots:
        print("[Task6] Displaying plot... close window to finish script.")
        plt.show()
    else:
        plt.close()

    segment_summary = df.groupby("Cluster")[features].mean().round(2)
    segment_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"))

    with open(os.path.join(OUTPUT_DIR, "segmentation_insights.txt"), "w", encoding="utf-8") as f:
        f.write("Task 6: Customer Segmentation Insights\n")
        f.write("=" * 38 + "\n\n")
        f.write("Features used: Annual Income (k$), Spending Score (1-100)\n")
        f.write("Number of clusters: 5\n\n")
        f.write("Cluster-wise average profile:\n")
        f.write(segment_summary.to_string() + "\n")

    print("[Task6] Cluster sizes:")
    print(df["Cluster"].value_counts().sort_index().to_string())
    print(f"[Task6] Completed. Segmentation outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 6 Customer segmentation")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot window; only save files.")
    args = parser.parse_args()
    main(show_plots=not args.no_show)
