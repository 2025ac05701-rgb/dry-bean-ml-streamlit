import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Dry Bean Classification - ML Model Comparison")

st.write(
    """
    This application evaluates different machine learning models
    on the UCI Dry Bean test dataset.
    """
)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "saved_models/logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            "saved_models/decision_tree.pkl"
        ),

        "KNN": joblib.load(
            "saved_models/knn.pkl"
        ),

        "Gaussian Naive Bayes": joblib.load(
            "saved_models/gaussian_nb.pkl"
        ),

        "Random Forest": joblib.load(
            "saved_models/random_forest.pkl"
        )
    }

    scaler = joblib.load(
        "saved_models/scaler.pkl"
    )

    label_encoder = joblib.load(
        "saved_models/label_encoder.pkl"
    )

    return models, scaler, label_encoder


models, scaler, label_encoder = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)


# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload the Dry Bean test CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Read uploaded CSV
    # --------------------------------------------------------

    test_data = pd.read_csv(uploaded_file)
     # -----------------------------
    # Clean column names
    # -----------------------------
    test_data.columns = test_data.columns.str.strip()

    # -----------------------------
    # Fix known column variations
    # -----------------------------
    test_data = test_data.rename(columns={
        "AspectRatio": "AspectRation",
        "Roundness": "roundness"
    })

    # -----------------------------
    # Get expected features
    # -----------------------------
    model_features = list(scaler.feature_names_in_)

    # -----------------------------
    # Check missing columns
    # -----------------------------
    missing_features = [
        col for col in model_features
        if col not in test_data.columns
    ]

    if missing_features:
        st.error(
            f"Missing feature columns: {missing_features}"
        )
        st.stop()

    # -----------------------------
    # Prepare model input
    # -----------------------------
    X_test_app = test_data[model_features]

    # -----------------------------
    # Scale
    # -----------------------------
    X_model = scaler.transform(X_test_app)
    st.success(
        "Test dataset uploaded successfully!"
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    st.subheader("Test Dataset Preview")

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )

    st.write(
        f"**Number of test samples:** {test_data.shape[0]}"
    )

    st.write(
        f"**Number of features:** {test_data.shape[1] - 1}"
    )


    # ========================================================
    # CHECK TARGET COLUMN
    # ========================================================

    target_column = "Class"

    if target_column not in test_data.columns:

        st.error(
            "The uploaded CSV must contain a 'Class' column."
        )

        st.stop()


    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    X_test_app = test_data.drop(
        columns=[target_column]
    )

    y_test_app = test_data[target_column]


    # ========================================================
    # CHECK FEATURE COLUMNS
    # ========================================================

    expected_features = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4"
    ]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X_test_app.columns
    ]


    if missing_features:

        st.error(
            f"Missing feature columns: {missing_features}"
        )

        st.stop()


    # Keep feature order exactly the same as training
    X_test_app = X_test_app[
        expected_features
    ]


    # ========================================================
    # ENCODE TARGET
    # ========================================================

    try:

        y_test_encoded = label_encoder.transform(
            y_test_app
        )

    except ValueError as e:

        st.error(
            "The uploaded dataset contains class labels "
            "that were not present during model training."
        )

        st.stop()


    # ========================================================
    # SELECT MODEL
    # ========================================================

    model = models[selected_model]


    # ========================================================
    # APPLY SCALING WHERE REQUIRED
    # ========================================================

    if selected_model in [
        "Logistic Regression",
        "KNN"
    ]:

        X_model = scaler.transform(
            X_test_app
        )

    else:

        X_model = X_test_app


    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = model.predict(
        X_model
    )

    y_prob = model.predict_proba(
        X_model
    )


    # ========================================================
    # EVALUATION METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test_encoded,
        y_pred
    )

    precision = precision_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0
    )

    auc = roc_auc_score(
        y_test_encoded,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )

    mcc = matthews_corrcoef(
        y_test_encoded,
        y_pred
    )


    # ========================================================
    # DISPLAY MODEL
    # ========================================================

    st.header("2. Selected Model")

    st.info(
        f"Currently selected model: **{selected_model}**"
    )


    # ========================================================
    # DISPLAY METRICS
    # ========================================================

    st.header("3. Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        st.metric(
            "Precision",
            f"{precision:.4f}"
        )


    with col2:

        st.metric(
            "AUC Score",
            f"{auc:.4f}"
        )

        st.metric(
            "Recall",
            f"{recall:.4f}"
        )


    with col3:

        st.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        st.metric(
            "MCC Score",
            f"{mcc:.4f}"
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.header("4. Confusion Matrix")

    cm = confusion_matrix(
        y_test_encoded,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    st.pyplot(fig)


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.header("5. Classification Report")

    report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    st.header("6. Prediction Results")

    prediction_results = X_test_app.copy()

    prediction_results["Actual Class"] = (
        label_encoder.inverse_transform(
            y_test_encoded
        )
    )

    prediction_results["Predicted Class"] = (
        label_encoder.inverse_transform(
            y_pred
        )
    )

    prediction_results["Correct"] = (
        prediction_results["Actual Class"]
        ==
        prediction_results["Predicted Class"]
    )

    st.dataframe(
        prediction_results,
        use_container_width=True
    )


else:

    st.info(
        "Please upload the test_data.csv file to begin evaluation."
    )
