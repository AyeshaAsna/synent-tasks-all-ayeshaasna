import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="CSV Dashboard", layout="wide")
st.title("Task 4: CSV to Interactive Dashboard")
st.write("Upload any CSV file and build dynamic charts.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for plotting.")
    else:
        st.subheader("Dynamic Charts")
        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox("Select X-axis", options=all_cols, index=0)
            y_col = st.selectbox("Select Y-axis (numeric)", options=numeric_cols, index=0)
            chart_type = st.selectbox("Chart type", ["Bar", "Scatter", "Histogram"])

        with col2:
            color_col = st.selectbox("Color by (optional)", options=["None"] + all_cols, index=0)
            color_arg = None if color_col == "None" else color_col

        if chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_arg, title=f"Bar Chart: {y_col} by {x_col}")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_arg, title=f"Scatter Plot: {y_col} vs {x_col}")
        else:
            fig = px.histogram(df, x=y_col, color=color_arg, title=f"Histogram: {y_col}")

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload a CSV file to start exploring.")
