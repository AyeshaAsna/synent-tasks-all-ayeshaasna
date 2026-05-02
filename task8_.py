import os
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Paths
BASE_DIR = r"D:\synent_tech_internship"
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "housing", "housing.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "tasks", "task8")


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    return {
        "model": name,
        "rmse": rmse,
        "r2": r2
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Task8] Loading dataset...")

    # ✅ Check dataset exists
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Dataset not found at: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    target = "median_house_value"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    X = df.drop(columns=[target])
    y = df[target]

    # Identify columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"[Task8] Numerical columns: {len(num_cols)}")
    print(f"[Task8] Categorical columns: {len(cat_cols)}")

    # ✅ Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[Task8] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Models
    linear_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Evaluate models
    results = [
        evaluate_model("Linear Regression", linear_pipeline, X_train, X_test, y_train, y_test),
        evaluate_model("Random Forest", rf_pipeline, X_train, X_test, y_train, y_test)
    ]

    results_df = pd.DataFrame(results).sort_values("rmse")

    # Save evaluation
    results_csv = os.path.join(OUTPUT_DIR, "model_evaluation.csv")
    results_df.to_csv(results_csv, index=False)

    print("\n[Task8] Model Evaluation:")
    print(results_df.to_string(index=False))

    # Select best model
    best_model_name = results_df.iloc[0]["model"]

    if best_model_name == "Linear Regression":
        best_model = linear_pipeline
    else:
        best_model = rf_pipeline

    # Train best model again on full train data
    best_model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)

    print(f"\n[Task8] Best Model: {best_model_name}")
    print(f"[Task8] Model saved at: {model_path}")

    # Sample prediction
    sample = X_test.iloc[[0]]
    prediction = best_model.predict(sample)

    print(f"\n[Task8] Sample Prediction: {prediction[0]:.2f}")
    print(f"[Task8] Actual Value: {y_test.iloc[0]:.2f}")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "task8_report.txt")

    with open(report_path, "w") as f:
        f.write("Task 8: Machine Learning Model Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {INPUT_PATH}\n\n")
        f.write("Model Evaluation:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Sample Prediction: {prediction[0]:.2f}\n")

    print(f"[Task8] Report saved at: {report_path}")
    print("\n[Task8] ✅ COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()