import os
import pandas as pd
import streamlit as st
from io import BytesIO
import joblib

# Define classify_data function


def classify_data(data, model):
    """
    Takes a DataFrame and applies the selected model, returning the labeled DataFrame.
    :param data: Uploaded data as a DataFrame.
    :param model: The classification model.
    """
    # Normalize column names
    data.columns = data.columns.str.strip().str.lower()

    # Define required columns
    required_columns = ['ts', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
                        'orig_bytes', 'resp_bytes', 'history', 'orig_pkts', 'orig_ip_bytes']

    # Map variations of column names
    column_mapping = {
        'ts': 'ts',
        'id.orig_h': 'id.orig_h',
        'id.orig_p': 'id.orig_p',
        'id.resp_h': 'id.resp_h',
        'id.resp_p': 'id.resp_p',
        'orig_bytes': 'orig_bytes',
        'resp_bytes': 'resp_bytes',
        'history': 'history',
        'orig_pkts': 'orig_pkts',
        'orig_ip_bytes': 'orig_ip_bytes'
    }

    # Rename columns
    data = data.rename(columns=column_mapping)

    # Check for missing required columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Clean and convert column values
    data = data[required_columns].apply(pd.to_numeric, errors='coerce')

    # Check for missing values
    if data.isnull().any().any():
        raise ValueError(
            "Some required columns contain invalid or missing values after cleaning."
        )

    # Predict labels
    predictions = model.predict(data)

    # Add predictions to the DataFrame
    class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    data['Prediction'] = [class_labels[int(p)] for p in predictions]

    return data

# Streamlit app


def main():
    st.title("CSV Classification with Model Management")

    # Sidebar for model management
    st.sidebar.title("Model Management")
    models_dir = "models"
    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # List all models
    available_models = [f for f in os.listdir(
        models_dir) if f.endswith(".pkl")]

    # Model uploader
    st.sidebar.subheader("Upload a New Model")
    model_upload = st.sidebar.file_uploader("Upload a .pkl file", type=["pkl"])
    if model_upload:
        model_path = os.path.join(models_dir, model_upload.name)
        with open(model_path, "wb") as f:
            f.write(model_upload.read())
        st.sidebar.success(
            f"Model '{model_upload.name}' uploaded successfully!")

    # Delete a model
    if available_models:
        st.sidebar.subheader("Delete a Model")
        model_to_delete = st.sidebar.selectbox(
            "Select a model to delete:", available_models)
        if st.sidebar.button("Delete Model"):
            os.remove(os.path.join(models_dir, model_to_delete))
            st.sidebar.success(
                f"Model '{model_to_delete}' deleted successfully!")

    # Select a model to use
    if available_models:
        st.sidebar.subheader("Select a Model for Classification")
        selected_model_name = st.sidebar.selectbox(
            "Choose a model:", available_models)
        selected_model_path = os.path.join(models_dir, selected_model_name)

        # Load the selected model
        model = joblib.load(selected_model_path)
        st.sidebar.success(f"Using model: {selected_model_name}")

        # File uploader for CSV
        st.subheader("Upload a CSV File for Classification")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            file_name = uploaded_file.name

            try:
                # Read CSV file
                data = pd.read_csv(uploaded_file, delimiter=';')
                st.write("Uploaded columns:", data.columns.tolist())

                # Classify data
                labeled_data = classify_data(data, model)

                # Display classified data
                st.success("Data classified successfully!")
                st.write("Classified Data:")
                st.dataframe(labeled_data)

                # Provide download link for labeled data
                output_file_name = 'labeled_' + file_name
                to_download = labeled_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Labeled CSV",
                    data=to_download,
                    file_name=output_file_name,
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.sidebar.error(
            "No models available. Please upload a model to proceed.")


if __name__ == "__main__":
    main()
