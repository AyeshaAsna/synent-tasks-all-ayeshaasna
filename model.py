import os
from pathlib import Path

import joblib
import pandas as pd
import qrcode
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / "datasets" / "WA_Fn-UseC_-Telco-Customer-Churn(task9).csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
EVALUATION_CSV_PATH = BASE_DIR / "evaluation.csv"
EVALUATION_REPORT_PATH = BASE_DIR / "evaluation_report.txt"
QR_PATH = BASE_DIR / "qr_code.png"


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # Required cleaning steps
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("Int64")

    # Keep rows with known label; feature missing values are handled in pipeline
    df = df.dropna(subset=["Churn"]).copy()
    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    if "customerID" in categorical_features:
        categorical_features.remove("customerID")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def save_evaluation(y_true: pd.Series, y_pred: pd.Series) -> None:
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df["accuracy"] = accuracy
    report_df.to_csv(EVALUATION_CSV_PATH, index=True)

    report_text = classification_report(y_true, y_pred)
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Customer Churn Model Evaluation\n")
        f.write("=" * 35 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report_text)


def save_qr_code() -> None:
    # Replace with actual deployed URL after Streamlit deployment.
    public_url = os.getenv("DEPLOYED_APP_URL", "https://your-app-name.streamlit.app")
    img = qrcode.make(public_url)
    img.save(QR_PATH)


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = load_and_clean_data(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    joblib.dump(pipeline, MODEL_PATH)
    save_evaluation(y_test, y_pred)
    save_qr_code()

    print(f"Model saved: {MODEL_PATH}")
    print(f"Evaluation CSV saved: {EVALUATION_CSV_PATH}")
    print(f"Evaluation report saved: {EVALUATION_REPORT_PATH}")
    print(f"QR code saved: {QR_PATH}")


if __name__ == "__main__":
    train()
