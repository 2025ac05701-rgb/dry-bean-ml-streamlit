# Dry Bean Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple
Machine Learning classification models for identifying different types
of dry beans based on their geometric and shape-related characteristics.

The project uses the Dry Bean Dataset and implements five Machine
Learning classification algorithms:

-   Logistic Regression
-   Decision Tree
-   k-Nearest Neighbors (kNN)
-   Naive Bayes
-   Random Forest (Ensemble)

The trained models are evaluated using Accuracy, AUC, Precision, Recall,
F1 Score, and Matthews Correlation Coefficient (MCC). A Streamlit
application is also developed to allow users to upload test data, select
a Machine Learning model, generate predictions, and view evaluation
results.

------------------------------------------------------------------------

## 2. Dataset Description

### Dataset Name

**Dry Bean Dataset**

### Dataset Description

The Dry Bean Dataset contains measurements extracted from images of dry
beans. These measurements describe the geometric properties and shape
characteristics of the beans and are used to classify beans into
different classes.

The dataset contains **16 input features** and one target variable.

### Input Features

    No. Feature
  ----- -----------------
      1 Area
      2 Perimeter
      3 MajorAxisLength
      4 MinorAxisLength
      5 AspectRatio
      6 Eccentricity
      7 ConvexArea
      8 EquivDiameter
      9 Extent
     10 Solidity
     11 Roundness
     12 Compactness
     13 ShapeFactor1
     14 ShapeFactor2
     15 ShapeFactor3
     16 ShapeFactor4

### Target Variable

The target variable is:

`Class`

The target represents the type/class of dry bean.

### Data Preprocessing

The following preprocessing steps were performed:

1.  Loaded the dataset.
2.  Separated input features and target variable.
3.  Encoded the target labels where required.
4.  Split the dataset into training and testing data.
5.  Applied feature scaling using `StandardScaler`.
6.  Used the same fitted scaler for transforming test data and Streamlit
    application input.

------------------------------------------------------------------------

## 3. GitHub Repository Link

**GitHub Repository:**\
https://github.com/2025ac05701-rgb/dry-bean-ml-streamlit.git

The repository contains the Streamlit application, trained Machine
Learning models, preprocessing objects, requirements file, and this
README file.

------------------------------------------------------------------------

## 4. Machine Learning Models Used

The following five Machine Learning models were trained and evaluated.

### 4.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates
the probability of a sample belonging to a particular class.

### 4.2 Decision Tree

Decision Tree is a tree-based classification algorithm that recursively
splits the data based on feature values to make classification
decisions.

### 4.3 k-Nearest Neighbors (kNN)

kNN classifies a sample based on the classes of its nearest training
samples.

### 4.4 Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based
on Bayes' theorem and the assumption of conditional independence between
features.

### 4.5 Random Forest (Ensemble)

Random Forest is an ensemble learning algorithm that combines multiple
decision trees to improve generalization and prediction performance.

------------------------------------------------------------------------

## 5. Evaluation Metrics

The models were evaluated using the following metrics:

-   **Accuracy:** Proportion of correctly classified samples.
-   **AUC:** Measures the ability of the model to distinguish between
    classes.
-   **Precision:** Measures how many of the samples predicted as a class
    actually belong to that class.
-   **Recall:** Measures how many of the actual samples of a class were
    correctly identified.
-   **F1 Score:** Harmonic mean of precision and recall.
-   **MCC:** Matthews Correlation Coefficient, which provides a balanced
    measure of classification quality.

------------------------------------------------------------------------

## 6. Model Comparison

The performance of all five Machine Learning models was evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|:---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8920 | 0.9334 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| kNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Random Forest (Ensemble) | 0.9203 | 0.9915 | 0.9205 | 0.9203 | 0.9203 | 0.9036 |
------------------------------------------------------------------------

## 7. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|:---|:---|
| **Logistic Regression** | Logistic Regression achieved the best overall performance with an Accuracy of **92.07%**, AUC of **0.9934**, F1 Score of **92.09%**, and MCC of **0.9042**. It achieved the highest scores across all six evaluation metrics, making it the best-performing model for this dataset. |
| **Decision Tree** | Decision Tree achieved an Accuracy of **89.20%** and an F1 Score of **89.16%**. Although it performed reasonably well, its performance was lower than Logistic Regression, kNN, and Random Forest across the major evaluation metrics. |
| **kNN** | kNN performed strongly with an Accuracy of **91.66%**, AUC of **0.9812**, and F1 Score of **91.68%**. Its performance was close to Logistic Regression and Random Forest, indicating that the feature space contains useful neighborhood-based patterns for classification. |
| **Naive Bayes** | Naive Bayes achieved the lowest overall classification performance, with an Accuracy of **76.39%**, F1 Score of **76.15%**, and MCC of **0.7154**. Although its AUC was relatively high at **0.9644**, its other metrics were significantly lower than the remaining models. |
| **Random Forest (Ensemble)** | Random Forest achieved excellent performance with an Accuracy of **92.03%**, AUC of **0.9915**, F1 Score of **92.03%**, and MCC of **0.9036**. It was a very close second to Logistic Regression and demonstrated strong and balanced classification performance. |
| **Overall Winner for the Dataset** | **Logistic Regression** is the overall winner because it achieved the highest Accuracy, AUC, Precision, Recall, F1 Score, and MCC among all five models. Random Forest was a very close second. |
------------------------------------------------------------------------

## 8. Overall Winner for the Dataset

### Overall Winner: **Logistic Regression**

Based on the evaluation results, **Logistic Regression is the overall
winner for this dataset**.

It achieved:

  Metric        Logistic Regression
  ----------- ---------------------
  Accuracy         **0.9206757253**
  AUC              **0.9933984517**
  Precision        **0.9215328435**
  Recall           **0.9206757253**
  F1 Score         **0.9208728071**
  MCC              **0.9041666635**

Logistic Regression achieved the highest Accuracy, AUC, Precision,
Recall, F1 Score, and MCC among the five evaluated models. Random Forest
was a very close second, with performance nearly identical to Logistic
Regression.

Therefore, based on the provided evaluation results, Logistic Regression
is selected as the best-performing model for the Dry Bean dataset.

------------------------------------------------------------------------

## 9. Streamlit Application

A Streamlit application was developed to provide an interactive
interface for the trained models.

### Application Features

-   Upload test data in CSV format.
-   Preview uploaded data.
-   Validate required feature columns.
-   Select a Machine Learning model.
-   Load the trained model from the `saved_models` directory.
-   Apply the saved feature scaler.
-   Generate predictions.
-   Display evaluation metrics when the `Class` column is available.
-   Compare model performance.

### Required Input Features

The uploaded CSV should contain the following columns:

``` text
Area
Perimeter
MajorAxisLength
MinorAxisLength
AspectRatio
Eccentricity
ConvexArea
EquivDiameter
Extent
Solidity
Roundness
Compactness
ShapeFactor1
ShapeFactor2
ShapeFactor3
ShapeFactor4
```

The `Class` column may also be included in the test CSV when evaluation
metrics are required.

------------------------------------------------------------------------

## 10. Project Structure

``` text
dry-bean-ml-streamlit/
│
├── app.py
├── requirements.txt
├── README.md
│
└── saved_models/
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

------------------------------------------------------------------------

## 11. Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Joblib
-   Streamlit
-   GitHub

------------------------------------------------------------------------

## 12. Installation and Execution

### Clone the Repository

``` bash
git clone https://github.com/2025ac05701-rgb/dry-bean-ml-streamlit.git
```

### Navigate to the Project Directory

``` bash
cd dry-bean-ml-streamlit
```

### Install Dependencies

``` bash
pip install -r requirements.txt
```

### Run the Streamlit Application

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 13. Deployment

The Streamlit application can be deployed using Streamlit Community
Cloud.

Deployment requirements:

1.  Push the project files to GitHub.
2.  Ensure `app.py` is present in the repository.
3.  Ensure `requirements.txt` contains the required dependencies.
4.  Ensure the `saved_models` folder contains all required `.pkl` files.
5.  Connect the GitHub repository to Streamlit Community Cloud.
6.  Select `app.py` as the application entry point.
7.  Deploy the application.

------------------------------------------------------------------------

## 14. Conclusion

This project demonstrates the application and comparison of five Machine
Learning classification algorithms on the Dry Bean Dataset.

Among the evaluated models, **Logistic Regression achieved the best
overall performance**, obtaining the highest Accuracy, AUC, Precision,
Recall, F1 Score, and MCC. Random Forest also performed exceptionally
well and was a close second.

The results demonstrate that relatively simple classification models can
achieve strong performance on the Dry Bean Dataset when appropriate
preprocessing and feature scaling are applied. The Streamlit application
provides an interactive way to use the trained models and evaluate
predictions on test data.

------------------------------------------------------------------------

## 15. Author

**Dhanashree Tare**

M.Tech in Artificial Intelligence and Machine Learning\
BITS Pilani

------------------------------------------------------------------------

## 16. Academic Project

This project is developed as part of an academic Machine Learning
assignment involving model implementation, evaluation, comparison, and
deployment using Streamlit.

