import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = r"D:\synent_tech_internship"
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "housing", "housing.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "tasks", "task8")


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    return {"model": name, "rmse": rmse, "r2": r2}


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[Task8] Loading housing dataset...")
    df = pd.read_csv(INPUT_PATH)
    target = "median_house_value"
    x = df.drop(columns=[target])
    y = df[target]

    num_cols = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = x.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"[Task8] Train size: {len(x_train)}, Test size: {len(x_test)}")

    linear_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])
    rf_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(n_estimators=200, random_state=42))]
    )

    results = [
        evaluate_model("Linear Regression", linear_pipeline, x_train, x_test, y_train, y_test),
        evaluate_model("Random Forest Regressor", rf_pipeline, x_train, x_test, y_train, y_test),
    ]
    print("[Task8] Model training and evaluation done.")
    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_evaluation.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "task8_report.txt"), "w", encoding="utf-8") as f:
        f.write("Task 8: Machine Learning Model Report\n")
        f.write("=" * 36 + "\n\n")
        f.write(f"Dataset: {INPUT_PATH}\n")
        f.write(f"Train size: {len(x_train)}, Test size: {len(x_test)}\n\n")
        f.write("Model evaluation:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nBest model selected by lowest RMSE.\n")

    print("[Task8] Evaluation table:")
    print(results_df.to_string(index=False))
    print(f"[Task8] Completed. ML outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
