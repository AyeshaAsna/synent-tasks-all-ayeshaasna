import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = r"D:\synent_tech_internship"
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "superstore", "Sample - Superstore.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "tasks", "task5")


def main(show_plots: bool) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("[Task5] Loading Superstore dataset...")
    # Superstore commonly ships with latin1/cp1252 encoding.
    df = pd.read_csv(INPUT_PATH, encoding="latin1")
    print(f"[Task5] Shape: {df.shape}")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

    # Monthly revenue trend.
    monthly_revenue = df.groupby("Month", as_index=False)["Sales"].sum().sort_values("Month")
    monthly_revenue.to_csv(os.path.join(OUTPUT_DIR, "monthly_revenue.csv"), index=False)
    print("[Task5] Monthly revenue computed.")

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly_revenue, x="Month", y="Sales", marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Revenue Trend")
    plt.tight_layout()
    monthly_path = os.path.join(OUTPUT_DIR, "monthly_revenue_trend.png")
    plt.savefig(monthly_path, dpi=150)
    print(f"[Task5] Saved: {monthly_path}")
    if not show_plots:
        plt.close()

    # Top-selling products.
    top_products = (
        df.groupby("Product Name", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    top_products.to_csv(os.path.join(OUTPUT_DIR, "top_10_products.csv"), index=False)
    print("[Task5] Top products computed.")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_products, y="Product Name", x="Sales", hue="Product Name", palette="Blues_r", legend=False)
    plt.title("Top 10 Selling Products by Sales")
    plt.tight_layout()
    top_path = os.path.join(OUTPUT_DIR, "top_products.png")
    plt.savefig(top_path, dpi=150)
    print(f"[Task5] Saved: {top_path}")
    if not show_plots:
        plt.close()

    # Profit analysis.
    profit_by_category = df.groupby("Category", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    profit_by_category.to_csv(os.path.join(OUTPUT_DIR, "profit_by_category.csv"), index=False)
    print("[Task5] Profit analysis computed.")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=profit_by_category, x="Category", y="Profit", hue="Category", palette="Set2", legend=False)
    plt.title("Profit by Category")
    plt.tight_layout()
    profit_path = os.path.join(OUTPUT_DIR, "profit_by_category.png")
    plt.savefig(profit_path, dpi=150)
    print(f"[Task5] Saved: {profit_path}")
    if show_plots:
        print("[Task5] Displaying plots... close plot windows to finish script.")
        plt.show()
    else:
        plt.close("all")

    with open(os.path.join(OUTPUT_DIR, "business_insights_report.md"), "w", encoding="utf-8") as f:
        f.write("# Task 5 Business Insights\n\n")
        f.write(f"- Total sales: {df['Sales'].sum():,.2f}\n")
        f.write(f"- Total profit: {df['Profit'].sum():,.2f}\n")
        f.write(f"- Best month by revenue: {monthly_revenue.loc[monthly_revenue['Sales'].idxmax(), 'Month']}\n")
        f.write(
            f"- Most profitable category: {profit_by_category.loc[profit_by_category['Profit'].idxmax(), 'Category']}\n"
        )

    print(f"[Task5] Total sales: {df['Sales'].sum():,.2f}")
    print(f"[Task5] Total profit: {df['Profit'].sum():,.2f}")
    print(f"[Task5] Completed. Sales analysis outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 5 Sales analysis")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows; only save files.")
    args = parser.parse_args()
    main(show_plots=not args.no_show)
