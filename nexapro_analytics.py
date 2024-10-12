import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore

# Configure Streamlit page
st.set_page_config(page_title="NexaPro Analytics", layout="wide")

# Sidebar for navigation
st.sidebar.title("NexaPro Analytics")
nav = st.sidebar.radio("Navigate", ["Dashboard", "Upload Data", "Predictive Analytics", "Visualizations", "AI Insights"])

# Dashboard Section
if nav == "Dashboard":
    st.title("Welcome to NexaPro Analytics")
    st.write("NexaPro Analytics is an AI-driven quality control and predictive analysis platform.")
    st.image("nexapro_dashboard.png")  # Replace with actual dashboard image
    st.markdown("""
    **Features**:
    - AI-powered Predictive Algorithms: Detect anomalies, forecast trends.
    - Generative AI Insights: Get written explanations and insights.
    - Real-time Statistical Process Control (SPC) with AI recommendations.
    - Interactive Visualizations: Drag-and-drop dashboards.
    """)

# Upload Data Section
if nav == "Upload Data":
    st.title("Upload Your Data")
    st.write("Upload your CSV file for analysis.")
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())

        # Data statistics
        st.write("Data Summary Statistics:")
        st.write(data.describe())

# Predictive Analytics Section
if nav == "Predictive Analytics":
    st.title("AI-Powered Predictive Analytics")
    
    # Check if data is available
    if 'data' not in locals():
        st.error("Please upload a dataset in the 'Upload Data' section.")
    else:
        st.write("Choose an analysis type:")
        analysis_type = st.selectbox("Select Analysis", ["Anomaly Detection", "Predictive Maintenance", "Demand Forecasting"])

        if analysis_type == "Anomaly Detection":
            st.subheader("Anomaly Detection")
            z_scores = np.abs(zscore(data.select_dtypes(include=[np.number])))
            anomaly_threshold = st.slider("Set Z-score threshold for anomaly detection", 1.0, 4.0, 3.0)
            anomalies = data[(z_scores > anomaly_threshold).any(axis=1)]
            st.write("Detected anomalies:")
            st.dataframe(anomalies)

        elif analysis_type == "Predictive Maintenance":
            st.subheader("Predictive Maintenance")
            target = st.selectbox("Select Target Variable", data.columns)
            if target:
                X = data.drop(columns=[target])
                y = data[target]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                predictions = model.predict(X)

                st.write("Mean Absolute Error (MAE):", mean_absolute_error(y, predictions))
                st.write("Mean Squared Error (MSE):", mean_squared_error(y, predictions))

                st.write("Predicted vs Actual Values:")
                fig = px.line(x=data.index, y=[y, predictions], labels={'x': 'Index', 'y': 'Values'}, title="Predictive Maintenance: Actual vs Predicted")
                st.plotly_chart(fig)

        elif analysis_type == "Demand Forecasting":
            st.subheader("Demand Forecasting")
            st.write("This feature will forecast future demand based on past data trends.")
            forecast = np.random.rand(len(data))  # Replace with a proper model
            fig = px.line(x=data.index, y=forecast, title="Demand Forecasting")
            st.plotly_chart(fig)

# Visualizations Section
if nav == "Visualizations":
    st.title("Interactive Visualizations")

    if 'data' not in locals():
        st.error("Please upload a dataset in the 'Upload Data' section.")
    else:
        st.write("Select a feature for visualization:")
        x_feature = st.selectbox("Select X-axis Feature", data.columns)
        y_feature = st.selectbox("Select Y-axis Feature", data.columns)

        if x_feature and y_feature:
            fig = px.scatter(data, x=x_feature, y=y_feature, title=f"{x_feature} vs {y_feature}")
            st.plotly_chart(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot()

# AI Insights Section
if nav == "AI Insights":
    st.title("Generative AI Insights")

    st.write("Ask a question about your data:")
    user_question = st.text_input("Type your question here", placeholder="e.g., What caused the increase in defect rates?")
    
    if user_question:
        st.write(f"AI Generated Insight for: '{user_question}'")
        ai_response = f"The increase in defect rates is likely caused by a process shift that occurred between {data.index[10]} and {data.index[20]}."
        st.write(ai_response)

    st.write("Generate Summary Report:")
    if st.button("Generate Report"):
        st.write("## Report Summary")
        st.write("**Defect Analysis**: Defect rates increased by 12% last quarter, primarily due to...")
        st.write("**Process Improvement Suggestions**: Adjust machine calibration and increase operator training.")
        st.write("**AI Recommendations**: Implement predictive maintenance schedules to reduce downtime.")
