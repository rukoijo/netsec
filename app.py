# app.py

import streamlit as st
from test_model import predict_emails

st.set_page_config(page_title="Email Classifier", layout="wide")
st.title("📧 Email Phishing Classifier")

st.markdown("Enter an email message below. The system will classify it as one of the following:")
st.markdown("- ✅ **Normal**\n- ⚠️ **Human-Generated Phishing**\n- 🤖 **AI-Generated Phishing**")

email_text = st.text_area("✉️ Email content", height=300)

if st.button("🔍 Analyze Email"):
    if email_text.strip():
        label, confidence = predict_emails([email_text])
        
        # 결과 출력
        st.success(f"Prediction: **{label}**")
        st.metric(label="Confidence", value=f"{confidence:.2%}")

        # 위험도 색상 bar
        if label == "Normal":
            st.progress(confidence)
        elif label == "Human-Phishing":
            st.warning("⚠️ Potential Human-written phishing email")
        else:
            st.error("🤖 Likely AI-generated phishing email")
    else:
        st.warning("Please enter email content.")
