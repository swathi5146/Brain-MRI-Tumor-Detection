import json
import streamlit as st
import pandas as pd
from PIL import Image

from predict import predict_image
from medical_report import generate_medical_report
from database import(save_prediction,get_prediction_history,get_total_predictions,get_latest_prediction,clear_history)

st.set_page_config(page_title="Brain MRI Tumor Detection", page_icon="🧠", layout="wide")
st.title("Brain MRI Tumor Detection")
st.sidebar.markdown("---")
st.sidebar.write("###Project Details")
st.sidebar.write("Transfer Learning : EfficientNetB0")
st.sidebar.write("Framework : TensorFlow")
st.sidebar.write("Frontend : Streamlit")
st.sidebar.write("Database : SQLite")
st.sidebar.write("LLM : Gemini AI")
st.sidebar.markdown("---")

st.sidebar.metric(label="Total Predictions", value=get_total_predictions())

latest= get_latest_prediction()
if latest:
    st.sidebar.metric(label="Laste Prediction", value=latest)

st.sidebar.markdown("---")

uploaded_file= st.file_uploader("choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image= Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    
    if st.button("🔍 Predict"):
        with st.spinner("Analysing image..."):
            predicted_class, confidence, prediction= predict_image(image)
            confidence_pct=confidence*100
        st.markdown("---")
        col1,col2=st.columns(2)
        col1.metric("Prediction", predicted_class)
        col1.metric("Confidence", f"{confidence_pct:.2f}%")
        
        with open("class_names.json") as f:
            class_names=json.load(f)

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": [p * 100 for p in prediction],
        }).sort_values("Probability (%)", ascending=False)
        st.dataframe(prob_df, use_container_width=True)

        with st.spinner("Generating AI medical report..."):
            report = generate_medical_report(predicted_class, confidence_pct)

        st.subheader("🩺 AI Medical Report")
        st.write(report)

        save_prediction(uploaded_file.name, predicted_class, confidence_pct, report)

        st.success("✅ Prediction Completed Successfully")

st.markdown("---")
st.subheader("📋 Prediction History")

history = get_prediction_history()

if not history:
    st.info("No prediction history available.")
else:
    history_df = pd.DataFrame(
        history,
        columns=["ID", "Date", "Image", "Prediction", "Confidence", "Medical Report"],
    )
    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
    )

    if st.button("🗑 Clear Prediction History"):
        clear_history()
        st.success("Prediction history cleared successfully.")
        st.rerun()

st.markdown("---")
st.markdown(
    """
    ### ℹ️ Disclaimer
    This application is intended only for **educational and research purposes**.
    The prediction is generated using an AI model and should **NOT** be considered
    a medical diagnosis. Always consult a qualified radiologist or neurologist
    for professional evaluation.
    """
)
























