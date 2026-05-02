import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
BASE_DIR = r"D:\synent_tech_internship"
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "market", "Market.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "tasks", "task7")


def main(show_plots: bool) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("[Task7] Loading market dataset...")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Expected stock time-series dataset not found at: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    required_cols = {"Date", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Task7 requires columns {required_cols} in {INPUT_PATH}, found: {set(df.columns)}")
    print(f"[Task7] Using dataset: {INPUT_PATH}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").copy()

    # Focus on one index for a clean trend/seasonality view.
    index_name = "NSEI"
    if "Index" in df.columns and (df["Index"] == index_name).any():
        ts = df[df["Index"] == index_name].copy()
    else:
        ts = df.copy()
        index_name = str(ts["Index"].iloc[0]) if "Index" in ts.columns else "Market"
    print(f"[Task7] Index selected: {index_name}, rows: {len(ts)}")

    ts["month"] = ts["Date"].dt.month
    ts["rolling_30"] = ts["Close"].rolling(window=30, min_periods=1).mean()

    # Trend plot.
    plt.figure(figsize=(11, 5))
    plt.plot(ts["Date"], ts["Close"], label="Close Price", alpha=0.7)
    plt.plot(ts["Date"], ts["rolling_30"], label="30-day Rolling Mean", linewidth=2)
    plt.title(f"Trend Analysis - {index_name}")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    trend_path = os.path.join(OUTPUT_DIR, "trend_analysis.png")
    plt.savefig(trend_path, dpi=150)
    print(f"[Task7] Saved: {trend_path}")
    if not show_plots:
        plt.close()

    # Seasonality view by month.
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=ts, x="month", y="Close")
    plt.title(f"Seasonality Detection by Month - {index_name}")
    plt.xlabel("Month")
    plt.tight_layout()
    seasonality_path = os.path.join(OUTPUT_DIR, "seasonality_by_month.png")
    plt.savefig(seasonality_path, dpi=150)
    print(f"[Task7] Saved: {seasonality_path}")
    if not show_plots:
        plt.close()

    # Optional forecast using simple linear regression over time index.
    ts = ts.reset_index(drop=True)
    ts["t"] = np.arange(len(ts))
    X = ts[["t"]]
    y = ts["Close"]
    model = LinearRegression()
    model.fit(X, y)
    print("[Task7] Trend model fitted.")

    future_steps = 30
    future_t = np.arange(len(ts), len(ts) + future_steps).reshape(-1, 1)
    forecast_values = model.predict(future_t)
    future_dates = pd.date_range(ts["Date"].iloc[-1] + pd.Timedelta(days=1), periods=future_steps, freq="D")

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast_Close": forecast_values})
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, "forecast_next_30_days.csv"), index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(ts["Date"], ts["Close"], label="Historical Close")
    plt.plot(forecast_df["Date"], forecast_df["Forecast_Close"], label="Forecast (Linear Regression)", color="red")
    plt.title(f"Forecast - {index_name}")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    forecast_path = os.path.join(OUTPUT_DIR, "forecast_plot.png")
    plt.savefig(forecast_path, dpi=150)
    print(f"[Task7] Saved: {forecast_path}")
    if show_plots:
        print("[Task7] Displaying plots... close plot windows to finish script.")
        plt.show()
    else:
        plt.close("all")

    with open(os.path.join(OUTPUT_DIR, "time_series_insights.txt"), "w", encoding="utf-8") as f:
        f.write("Task 7: Time Series Analysis Insights\n")
        f.write("=" * 36 + "\n\n")
        f.write(f"Index analyzed: {index_name}\n")
        f.write(f"Records used: {len(ts)}\n")
        f.write(f"Latest close price: {ts['Close'].iloc[-1]:.2f}\n")
        f.write(f"Average close price: {ts['Close'].mean():.2f}\n")

    print(f"[Task7] Forecast generated for {future_steps} days.")
    print(f"[Task7] Completed. Time-series outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 7 Time series analysis")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows; only save files.")
    args = parser.parse_args()
    main(show_plots=not args.no_show)
