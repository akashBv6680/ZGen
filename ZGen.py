import streamlit as st
import pandas as pd
import time
import os
import joblib
import requests
import yagmail
from threading import Thread
from pycaret.classification import *
from pycaret.regression import *
from pycaret.clustering import *
from pycaret.anomaly import *
from flask import Flask, request, jsonify

# === CONFIG ===
TOGETHER_API_KEY_1 = "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY"
TOGETHER_API_KEY_2 = "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"
EMAIL_ADDRESS = "akashvishnu6680@gmail.com"
EMAIL_PASSWORD = "swpe pwsx ypqo hgnk"
MODEL_NAME = "automl_model"

# === Streamlit Interface ===
st.title("Agentic AutoML App 🤖")

client_email = st.text_input("Enter your client's email address")
task = st.selectbox("Select ML Task", ["classification", "regression", "clustering", "anomaly"])
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

target = None
if task in ["classification", "regression"] and uploaded_file:
    df = pd.read_csv(uploaded_file)
    target = st.selectbox("Select target column", df.columns)

if st.button("Start AutoML Process"):
    if uploaded_file is None or (task in ["classification", "regression"] and not target) or not client_email:
        st.error("Please upload a dataset, select target, and enter client email.")
    else:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        with st.spinner("Running AutoML..."):
            # === Run PyCaret ===
            if task == "classification":
                setup(data=df, target=target, session_id=123, silent=True, html=False)
            elif task == "regression":
                setup(data=df, target=target, session_id=123, silent=True, html=False)
            elif task == "clustering":
                setup(data=df, session_id=123, html=False)
            elif task == "anomaly":
                setup(data=df, session_id=123, html=False)

            best_model = compare_models()
            tuned = tune_model(best_model)
            evaluate_model(tuned)
            interpret_model(tuned)
            save_model(tuned, MODEL_NAME)

            # === Send Email Notification ===
            try:
                yag = yagmail.SMTP(EMAIL_ADDRESS, EMAIL_PASSWORD)
                yag.send(to=client_email,
                         subject="✅ Your ML Model is Ready",
                         contents=f"Hi,\n\nYour {task} model has been trained, tuned and is ready to use.\n\nThanks,\nAgentic AI")
                st.success(f"Email sent to {client_email}!")
            except Exception as e:
                st.error(f"Failed to send email: {e}")

            st.success("Model training complete! 🎉")
            with open(f"{MODEL_NAME}.pkl", "rb") as f:
                st.download_button("Download Model", f, file_name=f"{MODEL_NAME}.pkl")


# === Optional Flask API Deployment ===
def start_flask_api():
    app = Flask(__name__)
    model = load_model(MODEL_NAME)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = pd.DataFrame(request.json)
        prediction = predict_model(model, data=data)
        return jsonify(prediction.to_dict(orient="records"))

    app.run(port=5000)

# === Optional Agentic AI Auto-Reply System ===
def agentic_auto_reply_loop(client_email):
    import imaplib
    import email

    def agentic_response(msg):
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY_1}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral-7b-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant responding to the client's questions about their ML model."},
                {"role": "user", "content": msg}
            ],
            "temperature": 0.7
        }
        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return "Sorry, I couldn't understand your message."

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("inbox")

    while True:
        status, messages = mail.search(None, f'(UNSEEN FROM "{client_email}")')
        if status == "OK":
            for num in messages[0].split():
                typ, data = mail.fetch(num, '(RFC822)')
                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        body = msg.get_payload(decode=True).decode()
                        reply = agentic_response(body)
                        yag = yagmail.SMTP(EMAIL_ADDRESS, EMAIL_PASSWORD)
                        yag.send(to=client_email, subject="🤖 Response to Your Message", contents=reply)
        time.sleep(60)

# === Button to Launch Flask API + Agentic Listener ===
if st.button("Start Flask API and Agentic AI Listener"):
    if os.path.exists(f"{MODEL_NAME}.pkl"):
        Thread(target=start_flask_api).start()
        Thread(target=agentic_auto_reply_loop, args=(client_email,)).start()
        st.success("Flask API and auto-response system started successfully!")
    else:
        st.error("Train the model before deploying the API.")
