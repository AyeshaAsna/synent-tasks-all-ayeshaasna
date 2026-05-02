import os
import pandas as pd


BASE_DIR = r"D:\synent_tech_internship"
INPUT_PATH = os.path.join(BASE_DIR, "datasets", "titanic.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "tasks", "task1")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "clean_titanic.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "task1_report.txt")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[Task1] Loading Titanic dataset...")
    df = pd.read_csv(INPUT_PATH)

    original_shape = df.shape
    missing_before = df.isna().sum()
    dup_before = int(df.duplicated().sum())
    print(f"[Task1] Original shape: {original_shape}")
    print(f"[Task1] Duplicate rows before: {dup_before}")

    # Remove duplicates first.
    df = df.drop_duplicates().copy()
    print("[Task1] Duplicates removed.")

    # Handle missing values by dtype.
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    # Convert data types where possible.
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median())
        df["Age"] = df["Age"].astype(float)
    if "Fare" in df.columns:
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce").fillna(df["Fare"].median())
    if "Passengerid" in df.columns:
        df["Passengerid"] = pd.to_numeric(df["Passengerid"], errors="coerce").astype("Int64")

    # Rename columns to standard snake_case.
    rename_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename_map)

    missing_after = df.isna().sum()
    dup_after = int(df.duplicated().sum())
    final_shape = df.shape
    print(f"[Task1] Final shape: {final_shape}")
    print(f"[Task1] Duplicate rows after: {dup_after}")

    df.to_csv(OUTPUT_CSV, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Task 1: Titanic Data Cleaning Report\n")
        f.write("=" * 42 + "\n\n")
        f.write(f"Input file: {INPUT_PATH}\n")
        f.write(f"Output file: {OUTPUT_CSV}\n\n")
        f.write(f"Original shape: {original_shape}\n")
        f.write(f"Final shape: {final_shape}\n\n")
        f.write(f"Duplicate rows before: {dup_before}\n")
        f.write(f"Duplicate rows after: {dup_after}\n\n")
        f.write("Missing values before cleaning:\n")
        f.write(missing_before.to_string() + "\n\n")
        f.write("Missing values after cleaning:\n")
        f.write(missing_after.to_string() + "\n")

    print("[Task1] Missing values after cleaning:")
    print(missing_after.to_string())
    print(f"[Task1] Completed. Cleaned dataset saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
