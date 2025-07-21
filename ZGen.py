import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import yagmail
import requests
from pycaret.classification import *
from pycaret.regression import *
from pycaret.clustering import *
from pycaret.anomaly import *
from flask import Flask, request, jsonify
from threading import Thread
from sklearn.preprocessing import LabelEncoder

# === CONFIG ===
EMAIL_ADDRESS = "akashvishnu6680@gmail.com"
EMAIL_PASSWORD = "swpe pwsx ypqo hgnk"
TOGETHER_API_KEY = "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY"
MODEL_NAME = "automl_model"
THEME_COLOR = "#0A9396"

# === STYLING ===
st.set_page_config(page_title="Agentic AutoML", layout="wide", page_icon="🤖")

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #F8F9FA;
        }}
        .big-font {{
            font-size:28px !important;
            color: {THEME_COLOR};
        }}
    </style>
    """, unsafe_allow_html=True
)

# === HEADER ===
st.markdown("<h1 class='big-font'>🤖 Agentic AutoML System</h1>", unsafe_allow_html=True)
st.markdown("Upload your dataset and let the agent decide the best ML approach.")

# === INPUTS ===
client_email = st.text_input("📧 Enter your client's email", placeholder="someone@example.com")
uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

df = None
target = None
task_type = None

# === READ & DISPLAY FILE ===
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(5))
        st.write("🧾 Column types:", df.dtypes)

        if not df.empty:
            possible_targets = df.columns[df.nunique() < 30].tolist()
            target = st.selectbox("🎯 Select target variable (or skip for unsupervised)", ["None"] + list(df.columns))
            if target != "None":
                if df[target].dtype == object or df[target].nunique() <= 10:
                    task_type = "classification"
                elif df[target].dtype in [np.float64, np.int64] and df[target].nunique() > 10:
                    task_type = "regression"
            else:
                task_type = st.selectbox("🔍 No target selected. Choose task manually:", ["clustering", "anomaly"])

            st.info(f"🧠 Detected Task Type: `{task_type}`")
        else:
            st.error("Empty dataset. Please upload a valid CSV.")

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

# === MAIN AUTOML BUTTON ===
if st.button("🚀 Run AutoML"):
    if df is not None and task_type:
        with st.spinner("Training the best model... please wait"):
            try:
                if task_type in ["classification", "regression"]:
                    setup(data=df, target=target, session_id=42, silent=True, html=False)
                else:
                    setup(data=df, session_id=42, silent=True, html=False)

                best_model = compare_models()
                tuned_model = tune_model(best_model)
                evaluate_model(tuned_model)
                interpret_model(tuned_model)
                save_model(tuned_model, MODEL_NAME)

                st.success("✅ Model training complete!")
                st.balloons()
                st.download_button("⬇️ Download Model", open(f"{MODEL_NAME}.pkl", "rb"), file_name=f"{MODEL_NAME}.pkl")

                if client_email:
                    try:
                        yag = yagmail.SMTP(EMAIL_ADDRESS, EMAIL_PASSWORD)
                        yag.send(
                            to=client_email,
                            subject="✅ Your Model is Ready",
                            contents=f"Hi,\n\nYour {task_type} model has been trained and is ready for use.\n\nThanks,\nAgentic AI"
                        )
                        st.success("📨 Client notified by email.")
                    except Exception as e:
                        st.warning(f"Failed to send email: {e}")

            except Exception as e:
                st.error(f"❌ AutoML process failed: {e}")
    else:
        st.error("Please upload a file and let the system detect a task.")

# === START API & AGENT ===
def start_flask_api():
    app = Flask(__name__)
    model = load_model(MODEL_NAME)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = pd.DataFrame(request.json)
        preds = predict_model(model, data=data)
        return jsonify(preds.to_dict(orient="records"))

    app.run(port=5000)

def start_agentic_listener(client_email):
    import imaplib
    import email

    def agentic_response(msg):
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "mistral-7b-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful ML assistant."},
                {"role": "user", "content": msg}
            ],
            "temperature": 0.7
        }
        res = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=body)
        return res.json()["choices"][0]["message"]["content"] if res.status_code == 200 else "I'm sorry, I couldn't generate a response."

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("inbox")

    while True:
        status, messages = mail.search(None, f'(UNSEEN FROM "{client_email}")')
        if status == "OK":
            for num in messages[0].split():
                typ, data = mail.fetch(num, '(RFC822)')
                for part in data:
                    if isinstance(part, tuple):
                        msg = email.message_from_bytes(part[1])
                        body = msg.get_payload(decode=True).decode()
                        reply = agentic_response(body)
                        yagmail.SMTP(EMAIL_ADDRESS, EMAIL_PASSWORD).send(
                            to=client_email,
                            subject="🤖 Agentic AI Response",
                            contents=reply
                        )
        time.sleep(60)

if st.button("🌐 Start API & Agentic AI"):
    if os.path.exists(f"{MODEL_NAME}.pkl") and client_email:
        Thread(target=start_flask_api).start()
        Thread(target=start_agentic_listener, args=(client_email,)).start()
        st.success("🚀 API and auto-responder started!")
    else:
        st.warning("Train the model and enter client email first.")
