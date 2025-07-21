import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
import shap
import smtplib
import ssl
from email.message import EmailMessage
import os

# --- Constants ---
TOGETHER_API_KEY_1 = "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY"
TOGETHER_API_KEY_2 = "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"

EMAIL_ADDRESS = "akashvishnu6680@gmail.com"
EMAIL_PASSWORD = "swpe pwsx ypqo hgnk"

# --- Page Config ---
st.set_page_config(page_title="Smart AutoML App", layout="wide")

# --- Title ---
st.title("🤖 Smart AutoML App with PyCaret")
st.caption("Upload your dataset, pick a target, and let AI build your model!")

# --- Upload ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Uploaded Successfully!")
    st.dataframe(df.head())

    # --- Target selection ---
    target = st.selectbox("🎯 Select your target column", df.columns)

    task_type = st.radio("🔍 Select Task Type", ["Classification", "Regression"])

    if st.button("🚀 Run AutoML"):
        st.spinner("Running PyCaret setup and model comparison...")
        st.toast("Running AutoML... please wait!")
        st.balloons()

        # Run PyCaret
        if task_type == "Classification":
            exp1 = setup(data=df, target=target, session_id=123, silent=True, verbose=False)
            best_model = compare_models()
            tuned_model = tune_model(best_model)
            evaluate_model(tuned_model)
            interpret_model(tuned_model)
        else:
            exp1 = pycaret.regression.setup(data=df, target=target, session_id=123, silent=True, verbose=False)
            best_model = pycaret.regression.compare_models()
            tuned_model = pycaret.regression.tune_model(best_model)
            pycaret.regression.evaluate_model(tuned_model)
            pycaret.regression.interpret_model(tuned_model)

        # Save model
        save_path = save_model(tuned_model, 'my_model')
        st.success("Model Trained and Saved!")

        # Deploy (simulated)
        st.info("Model deployment via Flask is ready. Run `flask run` separately.")

        # Email the client
        st.toast("Sending email to client...")

        def send_email():
            subject = "🎉 Your AI Model is Ready!"
            body = f"""
Hello,

Your machine learning model has been successfully trained and is ready to use!

Kind regards,  
Smart AutoML Agent  
"""
            msg = EmailMessage()
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = EMAIL_ADDRESS
            msg["Subject"] = subject
            msg.set_content(body)

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)

        try:
            send_email()
            st.success("Email sent to client! ✅")
        except Exception as e:
            st.error(f"Email failed: {e}")

        st.balloons()
