from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
DATA_PATH = BASE_DIR.parent.parent / "datasets" / "WA_Fn-UseC_-Telco-Customer-Churn(task9).csv"


@st.cache_data
def get_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run model.py first.")
    return joblib.load(MODEL_PATH)


def ui_input(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Customer Input")

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1397.0, step=1.0)

    sample_customer_id = "0000-STREAMLIT"
    if "customerID" in df.columns:
        sample_customer_id = str(df["customerID"].iloc[0])

    user_data = {
        "customerID": sample_customer_id,
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    return pd.DataFrame([user_data])


def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")
    st.title("Customer Churn Prediction App")
    st.write("Predict whether a telecom customer is likely to churn.")

    try:
        df = get_dataset()
        model = load_model()
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.info("Please ensure dataset exists and run model.py before launching app.")
        return

    input_df = ui_input(df)
    st.subheader("Input Data")
    st.dataframe(input_df)

    if st.button("Predict Churn"):
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            if int(pred) == 1:
                st.error("Prediction: Customer is likely to churn.")
            else:
                st.success("Prediction: Customer is likely to stay.")

            st.metric("Churn Probability", f"{float(proba) * 100:.2f}%")
            st.progress(int(np.clip(float(proba) * 100, 0, 100)))
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.warning("Please check inputs and ensure the saved model is valid.")


if __name__ == "__main__":
    main()
