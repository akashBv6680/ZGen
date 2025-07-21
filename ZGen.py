import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import requests
import yagmail
from email_validator import validate_email, EmailNotValidError
from pycaret.classification import *
from pycaret.regression import *
from pycaret.clustering import *
from pycaret.anomaly import *
from pycaret.arules import *
from pycaret.nlp import *
import base64
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Agentic AutoML App", layout="wide")
st.title("🤖 Agentic AutoML Platform")
st.markdown("""
<style>
    .reportview-container .markdown-text-container {
        font-family: 'Roboto';
        background-color: #f7f9fc;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and read successfully!")
        st.dataframe(df.head())

        target_column = st.selectbox("🎯 Select your target column (for supervised/NLP):", df.columns)
        ml_type = st.radio("🧪 Choose ML Task Type:", ("Auto-Detect", "Classification", "Regression", "Clustering", "Anomaly Detection", "Association Rules", "NLP"))
        client_email = st.text_input("📧 Enter your client’s email:")

        if st.button("🚀 Run AutoML"):
            with st.spinner("Training model(s)..."):
                try:
                    if ml_type == "Auto-Detect":
                        if df[target_column].nunique() <= 10 or df[target_column].dtype == 'object':
                            ml_type = "Classification"
                        else:
                            ml_type = "Regression"

                    if ml_type == "Classification":
                        setup(df, target=target_column, session_id=123, silent=True, verbose=False)
                        model = compare_models()
                    elif ml_type == "Regression":
                        setup(df, target=target_column, session_id=123, silent=True, verbose=False)
                        model = compare_models()
                    elif ml_type == "Clustering":
                        setup(df, session_id=123, silent=True, verbose=False)
                        model = create_model('kmeans')
                    elif ml_type == "Anomaly Detection":
                        setup(df, session_id=123, silent=True, verbose=False)
                        model = create_model('iforest')
                    elif ml_type == "Association Rules":
                        setup(df, transaction_id=df.columns[0], item_id=df.columns[1], session_id=123, silent=True, verbose=False)
                        model = create_model()
                    elif ml_type == "NLP":
                        setup(df, target=target_column, session_id=123, verbose=False)
                        model = create_model('lda')
                        st.subheader("📊 Wordcloud")
                        plot_model(model, plot='wordcloud', save=True)
                        st.image("Wordcloud.png")

                        st.subheader("📋 Topic Overview")
                        topics = assign_model(model)
                        st.dataframe(topics.head())

                        csv = topics.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Topics CSV", csv, "topics.csv", "text/csv")

                    if ml_type in ["Classification", "Regression"]:
                        tuned_model = tune_model(model)
                        evaluate_model(tuned_model)
                        interpret_model(tuned_model)
                        save_model(tuned_model, 'my_model')
                    else:
                        tuned_model = model

                    st.success("✅ Model complete!")
                    st.balloons()

                    if client_email:
                        try:
                            validate_email(client_email)
                            yag = yagmail.SMTP("akashvishnu6680@gmail.com", "swpe pwsx ypqo hgnk")
                            yag.send(to=client_email, subject="✅ Your AutoML model is ready!", contents="Your model has been trained and results are attached.")
                            st.success("📩 Email sent to client!")
                        except EmailNotValidError as e:
                            st.warning(f"Invalid email address: {e}")
                        except Exception as e:
                            st.warning(f"Failed to send email: {e}")
                except Exception as e:
                    st.error(f"❌ AutoML process failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ Failed to read file: {str(e)}")
else:
    st.info("📂 Please upload a dataset to get started.")
