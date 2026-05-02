from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / "datasets" / "WA_Fn-UseC_-Telco-Customer-Churn(task9).csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["ChurnBinary"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def plot_churn_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", color="#66c2a5")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "churn_distribution.png", dpi=200)
    plt.close()


def plot_monthly_charges_vs_churn(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", color="#8da0cb")
    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "monthlycharges_vs_churn.png", dpi=200)
    plt.close()


def plot_tenure_vs_churn(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="tenure", color="#fc8d62")
    plt.title("Tenure vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (months)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "tenure_vs_churn.png", dpi=200)
    plt.close()


def run_eda() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    sns.set_style("whitegrid")
    df = load_data()
    plot_churn_distribution(df)
    plot_monthly_charges_vs_churn(df)
    plot_tenure_vs_churn(df)
    print("EDA plots saved in task9 folder.")


if __name__ == "__main__":
    run_eda()
