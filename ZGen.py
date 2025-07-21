import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import requests
import yagmail

from email_validator import validate_email, EmailNotValidError

# PyCaret Modules (no NLP)
from pycaret.classification import *
from pycaret.regression import *
from pycaret.clustering import *
from pycaret.anomaly import *
from pycaret.arules import *

# 🎨 Streamlit page config
st.set_page_config(page_title="AutoML + Agentic AI", layout="wide")
st.title("🤖 AutoML with Agentic AI Integration")
st.markdown("""
This app auto-detects your ML task, trains the best model, explains it, and emails results to your client ✉️
""")

uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format only)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded successfully!")
        st.dataframe(df.head())

        task_type = None
        target_column = st.selectbox("🎯 Select target variable (or skip for unsupervised)", [None] + df.columns.tolist())

        if target_column:
            if df[target_column].dtype == 'object' or df[target_column].nunique() <= 20:
                task_type = 'classification'
                st.info("🧠 Detected Task Type: classification")
            else:
                task_type = 'regression'
                st.info("🧠 Detected Task Type: regression")
        else:
            st.info("🔍 No target selected. Using unsupervised mode (clustering/anomaly detection).")

        if st.button("🚀 Run AutoML"):
            with st.spinner("Running PyCaret... Please wait!"):
                if task_type == 'classification':
                    s = setup(df, target_column, session_id=123, silent=True, html=False)
                    best_model = compare_models()
                elif task_type == 'regression':
                    s = pycaret.regression.setup(df, target_column, session_id=123, silent=True, html=False)
                    best_model = pycaret.regression.compare_models()
                else:
                    s = setup(df, session_id=123, silent=True, html=False)
                    best_model = create_model('kmeans')

                tuned_model = tune_model(best_model)
                evaluate_model(tuned_model)
                interpret_model(tuned_model)
                save_model(tuned_model, 'best_model')

                st.success("✅ Model trained and saved as 'best_model.pkl'!")
                st.balloons()

        st.markdown("---")

        # Email Section
        st.subheader("📤 Send results to your client")
        client_email = st.text_input("Enter client's email address")

        if st.button("📧 Start API & Agentic AI"):
            try:
                validation = validate_email(client_email)
                client_email = validation.email

                # Simulated API call using Together API (Example)
                TOGETHER_API_KEY_1 = "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY"
                TOGETHER_API_KEY_2 = "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"
                headers = {"Authorization": f"Bearer {TOGETHER_API_KEY_1}"}
                data = {"prompt": "Summarize AutoML results in plain English.", "max_tokens": 100}
                response = requests.post("https://api.together.xyz/infer", json=data, headers=headers)

                if response.status_code == 200:
                    summary = response.json().get("output", "AutoML summary unavailable.")
                else:
                    summary = "Summary could not be generated."

                # Send Email
                yag = yagmail.SMTP("akashvishnu6680@gmail.com", "swpe pwsx ypqo hgnk")
                yag.send(to=client_email, subject="✅ AutoML Model Results", contents=summary)
                st.success(f"📬 Email sent to {client_email} successfully!")

            except EmailNotValidError as e:
                st.error(f"❌ Invalid email: {e}")
            except Exception as e:
                st.error(f"📛 Something went wrong: {e}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.warning("📥 Please upload a CSV file to begin.")
