import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import requests
import yagmail
from email_validator import validate_email, EmailNotValidError
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models, tune_model as classification_tune_model, evaluate_model as classification_evaluate_model, interpret_model as classification_interpret_model, save_model as classification_save_model, plot_model as classification_plot_model
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models, tune_model as regression_tune_model, evaluate_model as regression_evaluate_model, interpret_model as regression_interpret_model, save_model as regression_save_model, plot_model as regression_plot_model
from pycaret.clustering import setup as clustering_setup, create_model as clustering_create_model, plot_model as clustering_plot_model
from pycaret.anomaly import setup as anomaly_setup, create_model as anomaly_create_model, plot_model as anomaly_plot_model
from pycaret.arules import setup as arules_setup, create_model as arules_create_model
from pycaret.nlp import setup as nlp_setup, create_model as nlp_create_model, assign_model as nlp_assign_model, plot_model as nlp_plot_model
import base64
import matplotlib.pyplot as plt
from io import BytesIO

# Set page configuration
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
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for consistent UI behavior
if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_path' not in st.session_state:
    st.session_state.model_path = None
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False

# Helper function to capture matplotlib figures as bytes
def get_image_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

# Main application logic
uploaded_file = st.file_uploader("📄 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Reset state if a new file is uploaded
    if st.session_state.get('last_uploaded_file_name') != uploaded_file.name:
        st.session_state.df_uploaded = False
        st.session_state.model_trained = False
        st.session_state.model_path = None
        st.session_state.setup_complete = False
        st.session_state.last_uploaded_file_name = uploaded_file.name # Store for comparison

    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df # Store DataFrame in session state
        st.session_state.df_uploaded = True
        st.success("✅ File uploaded and read successfully!")
        st.dataframe(df.head())

        # UI elements for model selection and email, wrapped in a form
        with st.form("automl_form"):
            target_column = st.selectbox("🎯 Select your target column (for supervised/NLP):", df.columns)
            ml_type = st.radio("🧪 Choose ML Task Type:", ("Auto-Detect", "Classification", "Regression", "Clustering", "Anomaly Detection", "Association Rules", "NLP"))
            client_email = st.text_input("📧 Enter your client’s email (optional for result delivery):")

            submit_button = st.form_submit_button("🚀 Run AutoML")

            if submit_button:
                st.session_state.model_trained = False # Reset this state on new run
                st.session_state.model_path = None
                st.session_state.setup_complete = False

                with st.spinner("Training model(s)... This may take a while."):
                    try:
                        # Auto-detect logic
                        if ml_type == "Auto-Detect":
                            if target_column in df.columns: # Ensure target column exists
                                if df[target_column].nunique() <= 10 or df[target_column].dtype == 'object':
                                    ml_type = "Classification"
                                    st.info(f"Auto-detected ML Task: **Classification** (Target: '{target_column}' is categorical or has few unique values)")
                                else:
                                    ml_type = "Regression"
                                    st.info(f"Auto-detected ML Task: **Regression** (Target: '{target_column}' is numerical with many unique values)")
                            else:
                                st.warning("Cannot auto-detect ML type without a valid target column. Please select one or choose a type.")
                                # Fallback or exit
                                st.stop() # Stop execution if a critical condition isn't met

                        # PyCaret Setup (cached for efficiency)
                        @st.cache_resource(hash_funcs={pd.DataFrame: lambda _: uploaded_file.read()}) # Cache based on file content
                        def setup_pycaret(_df, _target=None, _transaction_id=None, _item_id=None, _ml_type=None):
                            if _ml_type in ["Classification", "Regression", "NLP"]:
                                st.info(f"Setting up PyCaret for {_ml_type} with target: '{_target}'...")
                                if _ml_type == "Classification":
                                    classification_setup(_df, target=_target, session_id=123, silent=True, verbose=False, html=False)
                                elif _ml_type == "Regression":
                                    regression_setup(_df, target=_target, session_id=123, silent=True, verbose=False, html=False)
                                elif _ml_type == "NLP":
                                    nlp_setup(_df, target=_target, session_id=123, verbose=False, html=False)
                            elif _ml_type == "Clustering":
                                st.info("Setting up PyCaret for Clustering...")
                                clustering_setup(_df, session_id=123, silent=True, verbose=False, html=False)
                            elif _ml_type == "Anomaly Detection":
                                st.info("Setting up PyCaret for Anomaly Detection...")
                                anomaly_setup(_df, session_id=123, silent=True, verbose=False, html=False)
                            elif _ml_type == "Association Rules":
                                if _transaction_id and _item_id:
                                    st.info(f"Setting up PyCaret for Association Rules with transaction_id: '{_transaction_id}' and item_id: '{_item_id}'...")
                                    arules_setup(_df, transaction_id=_transaction_id, item_id=_item_id, session_id=123, silent=True, verbose=False, html=False)
                                else:
                                    st.error("For Association Rules, please ensure the first two columns are suitable for Transaction ID and Item ID.")
                                    st.stop() # Stop if prerequisites not met
                            st.session_state.setup_complete = True
                            st.success("PyCaret Setup Complete!")

                        # Determine setup parameters
                        if ml_type == "Association Rules":
                            if len(df.columns) < 2:
                                st.error("For Association Rules, your dataset must have at least two columns.")
                                st.stop()
                            setup_pycaret(df, _transaction_id=df.columns[0], _item_id=df.columns[1], _ml_type=ml_type)
                        elif ml_type in ["Clustering", "Anomaly Detection"]:
                            setup_pycaret(df, _ml_type=ml_type)
                        else: # Supervised (Classification, Regression, NLP)
                            if target_column not in df.columns:
                                st.error(f"Target column '{target_column}' not found in the dataset. Please select a valid target column.")
                                st.stop()
                            setup_pycaret(df, _target=target_column, _ml_type=ml_type)


                        model = None
                        tuned_model = None

                        if ml_type == "Classification":
                            st.subheader("📚 Comparing Models (Classification)")
                            model = classification_compare_models()
                            st.write("Best Model:", model)
                            st.subheader("⚙️ Tuning Best Model")
                            tuned_model = classification_tune_model(model)
                            st.write("Tuned Model:", tuned_model)
                            st.subheader("📊 Model Evaluation Plots")
                            # PyCaret plots often generate matplotlib figures directly
                            try:
                                fig = classification_evaluate_model(tuned_model, plot_kwargs={'scale': 2}) # Use a dummy plot to trigger creation
                                if fig:
                                    st.pyplot(fig)
                                    st.download_button("Download Evaluation Plot", get_image_bytes(fig), "evaluation_plot.png", "image/png")
                                else:
                                    st.warning("Could not capture evaluation plots directly. PyCaret might be displaying them.")
                            except Exception as e:
                                st.warning(f"Failed to display evaluation plots: {e}")

                            st.subheader("🧠 Model Interpretation (SHAP/LIME)")
                            try:
                                fig = classification_interpret_model(tuned_model, plot='correlation') # Example plot
                                if fig:
                                    st.pyplot(fig)
                                    st.download_button("Download Interpretation Plot", get_image_bytes(fig), "interpretation_plot.png", "image/png")
                                else:
                                    st.warning("Could not capture interpretation plots directly. PyCaret might be displaying them.")
                            except Exception as e:
                                st.warning(f"Failed to display interpretation plots: {e}")

                        elif ml_type == "Regression":
                            st.subheader("📚 Comparing Models (Regression)")
                            model = regression_compare_models()
                            st.write("Best Model:", model)
                            st.subheader("⚙️ Tuning Best Model")
                            tuned_model = regression_tune_model(model)
                            st.write("Tuned Model:", tuned_model)
                            st.subheader("📊 Model Evaluation Plots")
                            try:
                                fig = regression_evaluate_model(tuned_model, plot_kwargs={'scale': 2})
                                if fig:
                                    st.pyplot(fig)
                                    st.download_button("Download Evaluation Plot", get_image_bytes(fig), "evaluation_plot.png", "image/png")
                                else:
                                    st.warning("Could not capture evaluation plots directly. PyCaret might be displaying them.")
                            except Exception as e:
                                st.warning(f"Failed to display evaluation plots: {e}")

                            st.subheader("🧠 Model Interpretation (SHAP/LIME)")
                            try:
                                fig = regression_interpret_model(tuned_model, plot='correlation')
                                if fig:
                                    st.pyplot(fig)
                                    st.download_button("Download Interpretation Plot", get_image_bytes(fig), "interpretation_plot.png", "image/png")
                                else:
                                    st.warning("Could not capture interpretation plots directly. PyCaret might be displaying them.")
                            except Exception as e:
                                st.warning(f"Failed to display interpretation plots: {e}")

                        elif ml_type == "Clustering":
                            st.subheader("📊 Clustering Model (K-Means)")
                            model = clustering_create_model('kmeans')
                            st.write("Created K-Means Model:", model)
                            st.subheader("Cluster Plots")
                            try:
                                # For clustering, plot_model usually takes the model and a plot type
                                fig = clustering_plot_model(model, plot='elbow', save=False) # save=False ensures it doesn't try to write to disk
                                if fig:
                                    st.pyplot(fig)
                                    st.download_button("Download Elbow Plot", get_image_bytes(fig), "elbow_plot.png", "image/png")
                                else:
                                    st.warning("Could not capture Elbow Plot.")

                                fig_silhouette = clustering_plot_model(model, plot='silhouette', save=False)
                                if fig_silhouette:
                                    st.pyplot(fig_silhouette)
                                    st.download_button("Download Silhouette Plot", get_image_bytes(fig_silhouette), "silhouette_plot.png", "image/png")
                                else:
                                    st.warning("Could not capture Silhouette Plot.")

                            except Exception as e:
                                st.warning(f"Failed to display clustering plots: {e}")
                            tuned_model = model # No tuning for basic clustering in PyCaret
                            
                        elif ml_type == "Anomaly Detection":
                            st.subheader("📈 Anomaly Detection Model (Isolation Forest)")
                            model = anomaly_create_model('iforest')
                            st.write("Created Isolation Forest Model:", model)
                            st.subheader("Anomaly Plots")
                            try:
                                fig = anomaly_plot_model(model, plot='tsne', save=False)
                                if fig:
                                    st.pyplot(fig)
                                    st.download_button("Download Anomaly Plot (t-SNE)", get_image_bytes(fig), "anomaly_tsne_plot.png", "image/png")
                                else:
                                    st.warning("Could not capture Anomaly Plot (t-SNE).")
                            except Exception as e:
                                st.warning(f"Failed to display anomaly plots: {e}")
                            tuned_model = model # No tuning for basic anomaly in PyCaret

                        elif ml_type == "Association Rules":
                            st.subheader("🛒 Association Rules Model")
                            model = arules_create_model()
                            st.dataframe(model)
                            st.download_button("📥 Download Association Rules CSV", model.to_csv(index=False).encode('utf-8'), "association_rules.csv", "text/csv")
                            tuned_model = model # No tuning for association rules

                        elif ml_type == "NLP":
                            st.subheader("📝 NLP Topic Model (LDA)")
                            model = nlp_create_model('lda')
                            st.write("Created LDA Model:", model)

                            st.subheader("📊 Wordcloud")
                            try:
                                fig = nlp_plot_model(model, plot='wordcloud', save=False)
                                if fig:
                                    st.pyplot(fig)
                                    st.download_button("Download Wordcloud", get_image_bytes(fig), "wordcloud.png", "image/png")
                                else:
                                    st.warning("Could not capture Wordcloud.")
                            except Exception as e:
                                st.warning(f"Failed to display wordcloud: {e}")

                            st.subheader("📋 Topic Overview")
                            topics = nlp_assign_model(model)
                            st.dataframe(topics.head())
                            csv = topics.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Topics CSV", csv, "topics.csv", "text/csv")
                            tuned_model = model # No tuning for basic NLP model

                        # Save model if it's a PyCaret trained model that can be saved
                        if tuned_model and ml_type in ["Classification", "Regression", "Clustering", "Anomaly Detection"]:
                            model_filename = 'my_pycaret_model.pkl'
                            st.info(f"Saving model to {model_filename}...")
                            # PyCaret's save_model saves to a file directly
                            if ml_type == "Classification":
                                classification_save_model(tuned_model, model_filename)
                            elif ml_type == "Regression":
                                regression_save_model(tuned_model, model_filename)
                            elif ml_type == "Clustering":
                                clustering_save_model(tuned_model, model_filename)
                            elif ml_type == "Anomaly Detection":
                                anomaly_save_model(tuned_model, model_filename)
                            st.session_state.model_path = model_filename
                            st.success(f"Model saved as '{model_filename}'")

                        st.success("✅ AutoML process complete!")
                        st.balloons()
                        st.session_state.model_trained = True

                        # Provide download link for the model
                        if st.session_state.model_path and os.path.exists(st.session_state.model_path):
                            with open(st.session_state.model_path, "rb") as f:
                                st.download_button(
                                    label="💾 Download Trained Model",
                                    data=f.read(),
                                    file_name=os.path.basename(st.session_state.model_path),
                                    mime="application/octet-stream"
                                )
                                # Clean up the saved model file after download button is presented
                                os.remove(st.session_state.model_path)
                                st.info("Model file has been provided for download and removed from server.")

                        # Email results
                        if client_email:
                            try:
                                validate_email(client_email)
                                # Consider attaching results like summary, plots, or the model itself.
                                # This part needs careful handling of attachments and email content.
                                # For a simple case:
                                email_content = "Your AutoML model training is complete! You can download the trained model and associated reports from the application."
                                # In a real scenario, you'd attach the model file and report images/CSVs
                                # For now, just sending a basic email without attachments for simplicity.
                                yag = yagmail.SMTP(os.environ.get("SENDER_EMAIL"), os.environ.get("SENDER_PASSWORD"))
                                # Ensure sender email and password are set as environment variables for security
                                yag.send(to=client_email, subject="✅ Your AutoML model is ready!", contents=email_content)
                                st.success("📩 Email sent to client!")
                            except EmailNotValidError as e:
                                st.warning(f"Invalid client email address provided: {e}")
                            except Exception as e:
                                st.warning(f"Failed to send email. Please check your email configuration or try again: {e}")
                                st.info("Ensure you have set SENDER_EMAIL and SENDER_PASSWORD environment variables for email functionality.")

                    except Exception as e:
                        st.error(f"❌ AutoML process failed: {str(e)}")
                        st.info("Please check your dataset, target column selection, and ML task type.")

    except Exception as e:
        st.error(f"❌ Failed to read file or process dataset: {str(e)}")
        st.info("Please ensure your file is a valid CSV and not corrupted.")
else:
    st.info("📂 Please upload a dataset to get started.")
    # Reset state when no file is uploaded
    st.session_state.df_uploaded = False
    st.session_state.model_trained = False
    st.session_state.model_path = None
    st.session_state.setup_complete = False
