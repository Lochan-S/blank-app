import streamlit as st

# ============================================
# IMPORT LIBRARIES
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
from sklearn.datasets import load_breast_cancer

# Data splitting and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ============================================
# EVALUATION METRICS
# ============================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# Reproducibility
RANDOM_STATE = 42


# ============================================
# LOAD WISCONSIN BREAST CANCER DATASET
# ============================================

cancer = load_breast_cancer()

# Create X from the 30 input features
X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

# Create y from the diagnosis/result
y = pd.Series(
    cancer.target,
    name="diagnosis"
)

# Create a complete DataFrame for inspection
df = pd.concat([X, y], axis=1)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

display(df.head())


# ============================================
# BASIC DATASET INFORMATION
# ============================================

print("Dataset shape:", df.shape)

print("\nFeature names:")
print(X.columns.tolist())

print("\nClass names:")
print(cancer.target_names)

print("\nClass distribution:")
print(y.value_counts())

print("\nDataset information:")
df.info()


# ============================================
# TARGET LABEL INFORMATION
# ============================================

label_names = {
    0: "Malignant",
    1: "Benign"
}

print("\nTarget encoding:")

for number, label in label_names.items():
    print(f"{number} = {label}")


# ============================================
# DATA QUALITY CHECK
# ============================================

print("\nDATASET QUALITY CHECK")
print("-" * 40)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Missing values
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Duplicate rows
print(f"Duplicate rows: {df.duplicated().sum()}")

# Infinite values
print(
    f"Infinite numerical values: "
    f"{np.isinf(df.select_dtypes(include=np.number)).sum().sum()}"
)

# Data types
print("\nData types:")
print(df.dtypes.value_counts())


# ============================================
# DESCRIPTIVE STATISTICS
# ============================================

print("\nDESCRIPTIVE STATISTICS")
display(df.describe())


# ============================================
# UNIQUE VALUES
# ============================================

print("\nUNIQUE VALUES PER COLUMN")
display(df.nunique())


# ============================================
# SPLIT DATA: 80% TRAIN / 20% TEST
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nDATA SPLIT")
print("-" * 40)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)


# ============================================
# STANDARDIZE FOR LOGISTIC REGRESSION
# ============================================

scaler = StandardScaler()

# IMPORTANT:
# The scaler learns the mean and standard deviation
# ONLY from the training data.
X_train_scaled = scaler.fit_transform(X_train)

# Apply the SAME training-based scaling to the test data.
X_test_scaled = scaler.transform(X_test)

print("\nSCALED DATA")
print("-" * 40)

print("Scaled training data:", X_train_scaled.shape)
print("Scaled testing data:", X_test_scaled.shape)

# ============================================
# LOGISTIC REGRESSION MODEL
# ============================================

# Create the Logistic Regression model
logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000
)

print("Logistic Regression model created.")


# ============================================
# TRAIN LOGISTIC REGRESSION
# ============================================

# Train ONLY using the training data
logistic_model.fit(
    X_train_scaled,
    y_train
)

print("Logistic Regression model trained successfully.")


# ============================================
# GET CLASS PREDICTIONS
# ============================================

# Predict the class for each test sample
logistic_predictions = logistic_model.predict(X_test_scaled)

print("\nClass predictions:")
print(logistic_predictions)

print("\nNumber of predictions:", len(logistic_predictions))


# ============================================
# GET PROBABILITY PREDICTIONS
# ============================================

# Get probabilities for both classes
logistic_probabilities = logistic_model.predict_proba(X_test_scaled)

print("\nProbability predictions:")
print(logistic_probabilities)

print("\nProbability array shape:", logistic_probabilities.shape)


# ============================================
# STORE BENIGN PROBABILITY
# ============================================

# Column 0 = Malignant
# Column 1 = Benign
logistic_benign_probability = logistic_probabilities[:, 1]

print("\nFirst 10 benign probabilities:")
print(logistic_benign_probability[:10])

# ============================================
# RANDOM FOREST MODEL
# ============================================

# Create the Random Forest model
random_forest_model = RandomForestClassifier(
    random_state=RANDOM_STATE
)

print("Random Forest model created.")


# ============================================
# TRAIN RANDOM FOREST
# ============================================

# Train ONLY using the training data
# Random Forest uses the original, unscaled features
random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest model trained successfully.")


# ============================================
# GET CLASS PREDICTIONS
# ============================================

# Predict the class for each sample in the SAME
# test set used by Logistic Regression
random_forest_predictions = random_forest_model.predict(X_test)

print("\nClass predictions:")
print(random_forest_predictions)

print("\nNumber of predictions:", len(random_forest_predictions))


# ============================================
# GET PROBABILITY PREDICTIONS
# ============================================

# Get probabilities for both classes
random_forest_probabilities = random_forest_model.predict_proba(X_test)

print("\nProbability predictions:")
print(random_forest_probabilities)

print("\nProbability array shape:", random_forest_probabilities.shape)


# ============================================
# STORE BENIGN PROBABILITY
# ============================================

# Column 0 = Malignant
# Column 1 = Benign
random_forest_benign_probability = random_forest_probabilities[:, 1]

print("\nFirst 10 benign probabilities:")
print(random_forest_benign_probability[:10])

# ============================================
# SUPPORT VECTOR MACHINE (SVM)
# ============================================

from sklearn.svm import SVC

# ============================================
# CREATE THE SVM MODEL
# ============================================

svm_model = SVC(
    probability=True,
    random_state=RANDOM_STATE
)

print("SVM model created.")


# ============================================
# TRAIN THE SVM
# ============================================

# Train ONLY on the scaled training data
# The test data is NOT used during training
svm_model.fit(
    X_train_scaled,
    y_train
)

print("SVM model trained successfully.")


# ============================================
# GET CLASS PREDICTIONS
# ============================================

# Make predictions on the SAME test cases
# used by Logistic Regression and Random Forest
svm_predictions = svm_model.predict(X_test_scaled)

print("\nSVM class predictions:")
print(svm_predictions)

print("\nNumber of predictions:", len(svm_predictions))


# ============================================
# GET PROBABILITY PREDICTIONS
# ============================================

# Get the probability of each class
svm_probabilities = svm_model.predict_proba(X_test_scaled)

print("\nSVM probability predictions:")
print(svm_probabilities)

print("\nProbability array shape:", svm_probabilities.shape)


# ============================================
# STORE BENIGN PROBABILITY
# ============================================

# Column 0 = Malignant
# Column 1 = Benign
svm_benign_probability = svm_probabilities[:, 1]

print("\nFirst 10 SVM benign probabilities:")
print(svm_benign_probability[:10])

# ============================================
# K-NEAREST NEIGHBORS (KNN) MODEL
# ============================================

from sklearn.neighbors import KNeighborsClassifier

# Create the KNN model
knn_model = KNeighborsClassifier(
    n_neighbors=5
)

print("KNN model created.")


# ============================================
# TRAIN KNN
# ============================================

# Train ONLY on the standardized training data
knn_model.fit(
    X_train_scaled,
    y_train
)

print("KNN model trained successfully.")


# ============================================
# GET CLASS PREDICTIONS
# ============================================

# Predict using the SAME test observations
knn_predictions = knn_model.predict(X_test_scaled)

print("\nKNN class predictions:")
print(knn_predictions)

print("\nNumber of KNN predictions:", len(knn_predictions))


# ============================================
# GET PROBABILITY PREDICTIONS
# ============================================

knn_probabilities = knn_model.predict_proba(X_test_scaled)

print("\nKNN probability predictions:")
print(knn_probabilities)

print("\nProbability array shape:", knn_probabilities.shape)


# ============================================
# STORE BENIGN PROBABILITY
# ============================================

# Column 0 = Malignant
# Column 1 = Benign
knn_benign_probability = knn_probabilities[:, 1]

print("\nFirst 10 KNN benign probabilities:")
print(knn_benign_probability[:10])

# ============================================
# CHECK THAT ALL MODELS USED THE SAME TEST SET
# ============================================

# Check that all models produced the same number of predictions
print("Number of test observations:")
print("Logistic Regression:", len(logistic_predictions))
print("Random Forest:", len(random_forest_predictions))
print("SVM:", len(svm_predictions))
print("KNN:", len(knn_predictions))
print("Actual test answers:", len(y_test))

# Verify that all prediction arrays have the same length
same_length = (
    len(logistic_predictions) == len(random_forest_predictions)
    == len(svm_predictions) == len(knn_predictions)
    == len(y_test)
)

print("\nAll prediction arrays have the same length:", same_length)

# Because every model received the SAME X_test / X_test_scaled,
# and the scaled version keeps the exact same row order,
# we can verify the predictions line up by observation number.
if same_length:
    print("✓ All four models made predictions for the same test observations.")
    print("✓ The prediction rows line up with y_test.")
else:
    print("✗ WARNING: The prediction arrays do not have the same length.")


# ============================================
# CALCULATE MODEL PERFORMANCE
# ============================================

# Logistic Regression
lr_accuracy = accuracy_score(y_test, logistic_predictions)
lr_precision = precision_score(y_test, logistic_predictions)
lr_recall = recall_score(y_test, logistic_predictions)
lr_roc_auc = roc_auc_score(y_test, logistic_benign_probability)
lr_incorrect = (logistic_predictions != y_test).sum()

# Random Forest
rf_accuracy = accuracy_score(y_test, random_forest_predictions)
rf_precision = precision_score(y_test, random_forest_predictions)
rf_recall = recall_score(y_test, random_forest_predictions)
rf_roc_auc = roc_auc_score(y_test, random_forest_benign_probability)
rf_incorrect = (random_forest_predictions != y_test).sum()

# SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_roc_auc = roc_auc_score(y_test, svm_benign_probability)
svm_incorrect = (svm_predictions != y_test).sum()

# KNN
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_roc_auc = roc_auc_score(y_test, knn_benign_probability)
knn_incorrect = (knn_predictions != y_test).sum()


# ============================================
# CREATE SIMPLE RESULTS TABLE
# ============================================

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN"
    ],
    "Accuracy": [
        lr_accuracy,
        rf_accuracy,
        svm_accuracy,
        knn_accuracy
    ],
    "Precision": [
        lr_precision,
        rf_precision,
        svm_precision,
        knn_precision
    ],
    "Recall": [
        lr_recall,
        rf_recall,
        svm_recall,
        knn_recall
    ],
    "ROC-AUC": [
        lr_roc_auc,
        rf_roc_auc,
        svm_roc_auc,
        knn_roc_auc
    ],
    "Incorrect Predictions": [
        lr_incorrect,
        rf_incorrect,
        svm_incorrect,
        knn_incorrect
    ]
})

# Display the table
display(model_results.round(4))


# ============================================
# IDENTIFY INCORRECT PREDICTIONS
# ============================================

# Create a table containing the real answer
# and each model's prediction
prediction_comparison = pd.DataFrame({
    "Test Observation": range(len(y_test)),
    "Actual": y_test.to_numpy(),
    "Logistic Regression": logistic_predictions,
    "Random Forest": random_forest_predictions,
    "SVM": svm_predictions,
    "KNN": knn_predictions
})

# Add columns showing whether each model was wrong
prediction_comparison["LR Wrong"] = (
    prediction_comparison["Logistic Regression"]
    != prediction_comparison["Actual"]
)

prediction_comparison["RF Wrong"] = (
    prediction_comparison["Random Forest"]
    != prediction_comparison["Actual"]
)

prediction_comparison["SVM Wrong"] = (
    prediction_comparison["SVM"]
    != prediction_comparison["Actual"]
)

prediction_comparison["KNN Wrong"] = (
    prediction_comparison["KNN"]
    != prediction_comparison["Actual"]
)


# ============================================
# DISPLAY ONLY INCORRECT PREDICTIONS
# ============================================

print("\nINCORRECT PREDICTIONS")
display(
    prediction_comparison[
        prediction_comparison[
            ["LR Wrong", "RF Wrong", "SVM Wrong", "KNN Wrong"]
        ].any(axis=1)
    ]
)

# ============================================
# CALCULATE MODEL DISAGREEMENT
# ============================================

# Absolute difference between the models'
# Benign probability predictions.

# 1. Logistic Regression vs Random Forest
lr_rf_disagreement = np.abs(
    logistic_benign_probability -
    random_forest_benign_probability
)

# 2. Logistic Regression vs SVM
lr_svm_disagreement = np.abs(
    logistic_benign_probability -
    svm_benign_probability
)

# 3. Logistic Regression vs KNN
lr_knn_disagreement = np.abs(
    logistic_benign_probability -
    knn_benign_probability
)

# 4. Random Forest vs SVM
rf_svm_disagreement = np.abs(
    random_forest_benign_probability -
    svm_benign_probability
)

# 5. Random Forest vs KNN
rf_knn_disagreement = np.abs(
    random_forest_benign_probability -
    knn_benign_probability
)

# 6. SVM vs KNN
svm_knn_disagreement = np.abs(
    svm_benign_probability -
    knn_benign_probability
)


# ============================================
# CHECK THE RESULTS
# ============================================

print("Disagreement calculations completed.")

print("\nNumber of observations in each pair:")
print("LR vs RF:", len(lr_rf_disagreement))
print("LR vs SVM:", len(lr_svm_disagreement))
print("LR vs KNN:", len(lr_knn_disagreement))
print("RF vs SVM:", len(rf_svm_disagreement))
print("RF vs KNN:", len(rf_knn_disagreement))
print("SVM vs KNN:", len(svm_knn_disagreement))


# ============================================
# CREATE DISAGREEMENT DATAFRAME
# ============================================

disagreement_data = pd.DataFrame({
    "Test Observation": range(len(y_test)),
    "Actual": y_test.to_numpy(),

    "LR vs RF": lr_rf_disagreement,
    "LR vs SVM": lr_svm_disagreement,
    "LR vs KNN": lr_knn_disagreement,
    "RF vs SVM": rf_svm_disagreement,
    "RF vs KNN": rf_knn_disagreement,
    "SVM vs KNN": svm_knn_disagreement
})


print("\nDISAGREEMENT DATAFRAME")
print("-" * 50)

display(disagreement_data.round(4))

# ============================================
# CREATE DISAGREEMENT DATAFRAME
# ============================================

# Create a sample/observation number so we can identify
# each test observation and keep the rows organized.
disagreement_data = pd.DataFrame({
    "Test Observation": range(len(y_test)),

    # The actual diagnosis for this test observation
    "Actual": y_test.to_numpy(),

    # Six model-pair disagreement values
    "LR vs RF": lr_rf_disagreement,
    "LR vs SVM": lr_svm_disagreement,
    "LR vs KNN": lr_knn_disagreement,
    "RF vs SVM": rf_svm_disagreement,
    "RF vs KNN": rf_knn_disagreement,
    "SVM vs KNN": svm_knn_disagreement
})


# ============================================
# ADD THE LARGEST DISAGREEMENT FOR EACH
# TEST OBSERVATION
# ============================================

# Look only at the six disagreement columns
disagreement_columns = [
    "LR vs RF",
    "LR vs SVM",
    "LR vs KNN",
    "RF vs SVM",
    "RF vs KNN",
    "SVM vs KNN"
]

# Find the largest disagreement among the six pairs
# for every test observation
disagreement_data["Maximum Disagreement"] = (
    disagreement_data[disagreement_columns].max(axis=1)
)

# Identify which model pair produced that maximum disagreement
disagreement_data["Most Disagreeing Pair"] = (
    disagreement_data[disagreement_columns].idxmax(axis=1)
)


# ============================================
# DISPLAY THE COMPLETE DISAGREEMENT TABLE
# ============================================

print("MODEL PAIR DISAGREEMENT FOR EVERY TEST OBSERVATION")
print("-" * 60)

display(
    disagreement_data.round(4)
)


# ============================================
# FIND OBSERVATIONS WITH THE HIGHEST
# OVERALL DISAGREEMENT
# ============================================

# Sort observations from highest to lowest
# based on their maximum disagreement
highest_disagreement = disagreement_data.sort_values(
    by="Maximum Disagreement",
    ascending=False
)

print("\nOBSERVATIONS WITH THE HIGHEST DISAGREEMENT")
print("-" * 60)

# Show the 10 observations with the highest disagreement
display(
    highest_disagreement[
        [
            "Test Observation",
            "Actual",
            "Maximum Disagreement",
            "Most Disagreeing Pair"
        ] + disagreement_columns
    ].head(10).round(4)
)

# ============================================
# ADD MODEL PREDICTIONS TO THE DATAFRAME
# ============================================

disagreement_data["Logistic Regression"] = logistic_predictions
disagreement_data["Random Forest"] = random_forest_predictions
disagreement_data["SVM"] = svm_predictions
disagreement_data["KNN"] = knn_predictions


# ============================================
# CALCULATE INDIVIDUAL MODEL UNCERTAINTY
# ============================================

# Your formula:
# Uncertainty = 1 - 2(P - 0.5)
#
# P = probability of the positive class (Benign)

disagreement_data["LR Uncertainty"] = (
    1 - 2 * np.abs(logistic_benign_probability - 0.5)
)

disagreement_data["RF Uncertainty"] = (
    1 - 2 * np.abs(random_forest_benign_probability - 0.5)
)

disagreement_data["SVM Uncertainty"] = (
    1 - 2 * np.abs(svm_benign_probability - 0.5)
)

disagreement_data["KNN Uncertainty"] = (
    1 - 2 * np.abs(knn_benign_probability - 0.5)
)


# ============================================
# IDENTIFY WRONG PREDICTIONS
# ============================================

# 0 = correct
# 1 = wrong

disagreement_data["LR Wrong"] = (
    disagreement_data["Logistic Regression"]
    != disagreement_data["Actual"]
).astype(int)

disagreement_data["RF Wrong"] = (
    disagreement_data["Random Forest"]
    != disagreement_data["Actual"]
).astype(int)

disagreement_data["SVM Wrong"] = (
    disagreement_data["SVM"]
    != disagreement_data["Actual"]
).astype(int)

disagreement_data["KNN Wrong"] = (
    disagreement_data["KNN"]
    != disagreement_data["Actual"]
).astype(int)


# ============================================
# CHECK THAT EVERYTHING IS ALIGNED
# ============================================

print("DATA ALIGNMENT CHECK")
print("-" * 50)

print("Number of test observations:", len(y_test))
print("Rows in dataframe:", len(disagreement_data))

print("\nPrediction lengths:")
print("Logistic Regression:", len(logistic_predictions))
print("Random Forest:", len(random_forest_predictions))
print("SVM:", len(svm_predictions))
print("KNN:", len(knn_predictions))

print("\nAll rows aligned:",
      len(disagreement_data) == len(y_test) ==
      len(logistic_predictions) ==
      len(random_forest_predictions) ==
      len(svm_predictions) ==
      len(knn_predictions))


# ============================================
# SHOW THE UPDATED DATAFRAME
# ============================================

print("\nUPDATED ANALYSIS DATAFRAME")
print("-" * 50)

display(disagreement_data.round(4))

# ============================================
# UNCERTAINTY + DISAGREEMENT ANALYSIS
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================
# 1. CALCULATE INDIVIDUAL MODEL UNCERTAINTY
# ============================================

# Your formula:
#
# Uncertainty = 1 - 2(P - 0.5)
#
# P = probability of the positive class (Benign)
#
# 0 = very confident in Benign (P = 1)
# 1 = uncertain (P = 0.5)
# 2 = very confident in Malignant (P = 0)
#
# We are using the formula exactly as specified.

logistic_uncertainty = (
    1 - 2 * np.abs(logistic_benign_probability - 0.5)
)

random_forest_uncertainty = (
    1 - 2 * np.abs(random_forest_benign_probability - 0.5)
)

svm_uncertainty = (
    1 - 2 * np.abs(svm_benign_probability - 0.5)
)

knn_uncertainty = (
    1 - 2 * np.abs(knn_benign_probability - 0.5)
)


# ============================================
# 2. ADD UNCERTAINTY TO EXISTING DATAFRAME
# ============================================

disagreement_data["LR Uncertainty"] = logistic_uncertainty
disagreement_data["RF Uncertainty"] = random_forest_uncertainty
disagreement_data["SVM Uncertainty"] = svm_uncertainty
disagreement_data["KNN Uncertainty"] = knn_uncertainty


# ============================================
# CHECK THE DATAFRAME
# ============================================

print("DATAFRAME WITH UNCERTAINTY VALUES")
print("-" * 60)

display(disagreement_data.round(4))


# ============================================
# 3. CREATE WRONG-PREDICTION INDICATORS
# ============================================

# 0 = correct prediction
# 1 = incorrect prediction

disagreement_data["LR Wrong"] = (
    disagreement_data["Logistic Regression"]
    != disagreement_data["Actual"]
).astype(int)

disagreement_data["RF Wrong"] = (
    disagreement_data["Random Forest"]
    != disagreement_data["Actual"]
).astype(int)

disagreement_data["SVM Wrong"] = (
    disagreement_data["SVM"]
    != disagreement_data["Actual"]
).astype(int)

disagreement_data["KNN Wrong"] = (
    disagreement_data["KNN"]
    != disagreement_data["Actual"]
).astype(int)


# ============================================
# 4. INDIVIDUAL UNCERTAINTY VS PREDICTION ERROR
# ============================================

# Four scatter plots:
# X-axis = model uncertainty
# Y-axis = whether prediction was wrong
#
# 0 = correct
# 1 = wrong

models_uncertainty_error = [
    ("Logistic Regression", "LR Uncertainty", "LR Wrong"),
    ("Random Forest", "RF Uncertainty", "RF Wrong"),
    ("SVM", "SVM Uncertainty", "SVM Wrong"),
    ("KNN", "KNN Uncertainty", "KNN Wrong")
]

for model_name, uncertainty_column, error_column in models_uncertainty_error:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        disagreement_data[uncertainty_column],
        disagreement_data[error_column],
        alpha=0.7
    )

    plt.xlabel("Uncertainty")
    plt.ylabel("Prediction Wrong (0 = Correct, 1 = Wrong)")
    plt.title(f"{model_name}: Individual Uncertainty vs Prediction Error")

    plt.yticks([0, 1], ["Correct", "Wrong"])
    plt.grid(True, alpha=0.3)

    plt.show()


# ============================================
# 5. CREATE "AT LEAST ONE MODEL WRONG"
# ============================================

# For each model pair, determine whether at least
# one of the two models made an incorrect prediction.

disagreement_data["LR-RF At Least One Wrong"] = (
    (disagreement_data["LR Wrong"] == 1) |
    (disagreement_data["RF Wrong"] == 1)
).astype(int)

disagreement_data["LR-SVM At Least One Wrong"] = (
    (disagreement_data["LR Wrong"] == 1) |
    (disagreement_data["SVM Wrong"] == 1)
).astype(int)

disagreement_data["LR-KNN At Least One Wrong"] = (
    (disagreement_data["LR Wrong"] == 1) |
    (disagreement_data["KNN Wrong"] == 1)
).astype(int)

disagreement_data["RF-SVM At Least One Wrong"] = (
    (disagreement_data["RF Wrong"] == 1) |
    (disagreement_data["SVM Wrong"] == 1)
).astype(int)

disagreement_data["RF-KNN At Least One Wrong"] = (
    (disagreement_data["RF Wrong"] == 1) |
    (disagreement_data["KNN Wrong"] == 1)
).astype(int)

disagreement_data["SVM-KNN At Least One Wrong"] = (
    (disagreement_data["SVM Wrong"] == 1) |
    (disagreement_data["KNN Wrong"] == 1)
).astype(int)


# ============================================
# 6. DISAGREEMENT VS PREDICTION ERROR
# ============================================

# X-axis = disagreement
# Y-axis = whether at least one model was wrong

disagreement_error_pairs = [
    ("LR vs RF", "LR-RF At Least One Wrong"),
    ("LR vs SVM", "LR-SVM At Least One Wrong"),
    ("LR vs KNN", "LR-KNN At Least One Wrong"),
    ("RF vs SVM", "RF-SVM At Least One Wrong"),
    ("RF vs KNN", "RF-KNN At Least One Wrong"),
    ("SVM vs KNN", "SVM-KNN At Least One Wrong")
]

for pair_name, error_column in disagreement_error_pairs:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        disagreement_data[pair_name],
        disagreement_data[error_column],
        alpha=0.7
    )

    plt.xlabel("Model Disagreement")
    plt.ylabel("At Least One Model Wrong (0 = No, 1 = Yes)")
    plt.title(f"{pair_name}: Disagreement vs Prediction Error")

    plt.yticks([0, 1], ["Neither Wrong", "At Least One Wrong"])
    plt.grid(True, alpha=0.3)

    plt.show()


# ============================================
# 7. ERROR RATE BY UNCERTAINTY LEVEL
# ============================================

uncertainty_columns = [
    ("Logistic Regression", "LR Uncertainty", "LR Wrong"),
    ("Random Forest", "RF Uncertainty", "RF Wrong"),
    ("SVM", "SVM Uncertainty", "SVM Wrong"),
    ("KNN", "KNN Uncertainty", "KNN Wrong")
]

for model_name, uncertainty_column, error_column in uncertainty_columns:

    # Rank the observations first.
    # This prevents repeated uncertainty values from
    # causing qcut() to fail.
    ranked_uncertainty = disagreement_data[
        uncertainty_column
    ].rank(method="first")

    # Divide the observations into four groups
    groups = pd.qcut(
        ranked_uncertainty,
        q=4,
        labels=[
            "Lowest",
            "Low-Medium",
            "Medium-High",
            "Highest"
        ]
    )

    # Calculate the error rate in each group
    error_rate = disagreement_data.groupby(
        groups,
        observed=False
    )[error_column].mean()

    # Graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        error_rate.index.astype(str),
        error_rate.values,
        marker="o"
    )

    plt.xlabel("Uncertainty Level")
    plt.ylabel("Error Rate")
    plt.title(f"{model_name}: Error Rate by Uncertainty Level")

    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\n{model_name} - Error Rate by Uncertainty Level")
    print(error_rate.round(4))


# ============================================
# 8. ERROR RATE BY DISAGREEMENT LEVEL
# ============================================

disagreement_error_pairs = [
    ("LR vs RF", "LR-RF At Least One Wrong"),
    ("LR vs SVM", "LR-SVM At Least One Wrong"),
    ("LR vs KNN", "LR-KNN At Least One Wrong"),
    ("RF vs SVM", "RF-SVM At Least One Wrong"),
    ("RF vs KNN", "RF-KNN At Least One Wrong"),
    ("SVM vs KNN", "SVM-KNN At Least One Wrong")
]

for pair_name, error_column in disagreement_error_pairs:

    # Rank the observations first to handle repeated
    # disagreement values.
    ranked_disagreement = disagreement_data[
        pair_name
    ].rank(method="first")

    # Divide observations into four groups
    groups = pd.qcut(
        ranked_disagreement,
        q=4,
        labels=[
            "Lowest",
            "Low-Medium",
            "Medium-High",
            "Highest"
        ]
    )

    # Calculate error rate for each group
    error_rate = disagreement_data.groupby(
        groups,
        observed=False
    )[error_column].mean()

    # Graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        error_rate.index.astype(str),
        error_rate.values,
        marker="o"
    )

    plt.xlabel("Disagreement Level")
    plt.ylabel("Error Rate")
    plt.title(f"{pair_name}: Error Rate by Disagreement Level")

    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\n{pair_name} - Error Rate by Disagreement Level")
    print(error_rate.round(4))
# ============================================
# 9. UNCERTAINTY VS DISAGREEMENT
# ============================================

# Compare individual model uncertainty with the
# disagreement between models.

uncertainty_disagreement_pairs = [
    ("Logistic Regression", "LR Uncertainty", "LR vs RF", "Random Forest"),
    ("Logistic Regression", "LR Uncertainty", "LR vs SVM", "SVM"),
    ("Logistic Regression", "LR Uncertainty", "LR vs KNN", "KNN"),
    ("Random Forest", "RF Uncertainty", "LR vs RF", "Logistic Regression"),
    ("Random Forest", "RF Uncertainty", "RF vs SVM", "SVM"),
    ("Random Forest", "RF Uncertainty", "RF vs KNN", "KNN"),
    ("SVM", "SVM Uncertainty", "LR vs SVM", "Logistic Regression"),
    ("SVM", "SVM Uncertainty", "RF vs SVM", "Random Forest"),
    ("SVM", "SVM Uncertainty", "SVM vs KNN", "KNN"),
    ("KNN", "KNN Uncertainty", "LR vs KNN", "Logistic Regression"),
    ("KNN", "KNN Uncertainty", "RF vs KNN", "Random Forest"),
    ("KNN", "KNN Uncertainty", "SVM vs KNN", "SVM")
]

for model_name, uncertainty_column, disagreement_column, other_model in uncertainty_disagreement_pairs:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        disagreement_data[uncertainty_column],
        disagreement_data[disagreement_column],
        alpha=0.7
    )

    plt.xlabel(f"{model_name} Uncertainty")
    plt.ylabel(f"{disagreement_column} Disagreement")
    plt.title(
        f"{model_name} Uncertainty vs {model_name} vs {other_model} Disagreement"
    )

    plt.grid(True, alpha=0.3)

    plt.show()


# ============================================
# 10. FINAL DATAFRAME CHECK
# ============================================

print("\nFINAL ANALYSIS DATAFRAME")
print("-" * 60)

print("Rows:", disagreement_data.shape[0])
print("Columns:", disagreement_data.shape[1])

display(disagreement_data.round(4))

# ============================================
# ROC-AUC: UNCERTAINTY VS DISAGREEMENT
# ============================================

from sklearn.metrics import roc_auc_score


# ============================================
# CALCULATE ROC-AUC VALUES
# ============================================

# Individual model uncertainty
lr_uncertainty_auc = roc_auc_score(
    disagreement_data["LR Wrong"],
    disagreement_data["LR Uncertainty"]
)

rf_uncertainty_auc = roc_auc_score(
    disagreement_data["RF Wrong"],
    disagreement_data["RF Uncertainty"]
)

svm_uncertainty_auc = roc_auc_score(
    disagreement_data["SVM Wrong"],
    disagreement_data["SVM Uncertainty"]
)

knn_uncertainty_auc = roc_auc_score(
    disagreement_data["KNN Wrong"],
    disagreement_data["KNN Uncertainty"]
)


# Model-pair disagreement
lr_rf_auc = roc_auc_score(
    disagreement_data["LR-RF At Least One Wrong"],
    disagreement_data["LR vs RF"]
)

lr_svm_auc = roc_auc_score(
    disagreement_data["LR-SVM At Least One Wrong"],
    disagreement_data["LR vs SVM"]
)

lr_knn_auc = roc_auc_score(
    disagreement_data["LR-KNN At Least One Wrong"],
    disagreement_data["LR vs KNN"]
)

rf_svm_auc = roc_auc_score(
    disagreement_data["RF-SVM At Least One Wrong"],
    disagreement_data["RF vs SVM"]
)

rf_knn_auc = roc_auc_score(
    disagreement_data["RF-KNN At Least One Wrong"],
    disagreement_data["RF vs KNN"]
)

svm_knn_auc = roc_auc_score(
    disagreement_data["SVM-KNN At Least One Wrong"],
    disagreement_data["SVM vs KNN"]
)


# ============================================
# PRINT ALL 10 ROC-AUC VALUES
# ============================================

print("ROC-AUC RESULTS")
print("-" * 50)

print(f"Logistic Regression uncertainty: {lr_uncertainty_auc:.4f}")
print(f"Random Forest uncertainty:       {rf_uncertainty_auc:.4f}")
print(f"SVM uncertainty:                 {svm_uncertainty_auc:.4f}")
print(f"KNN uncertainty:                  {knn_uncertainty_auc:.4f}")

print()

print(f"LR vs RF disagreement:            {lr_rf_auc:.4f}")
print(f"LR vs SVM disagreement:           {lr_svm_auc:.4f}")
print(f"LR vs KNN disagreement:           {lr_knn_auc:.4f}")
print(f"RF vs SVM disagreement:           {rf_svm_auc:.4f}")
print(f"RF vs KNN disagreement:           {rf_knn_auc:.4f}")
print(f"SVM vs KNN disagreement:          {svm_knn_auc:.4f}")

# ============================================
# BAR GRAPH: ROC-AUC COMPARISON
# ============================================

# Names of the 10 measurements
labels = [
    "LR uncertainty",
    "RF uncertainty",
    "SVM uncertainty",
    "KNN uncertainty",
    "LR-RF disagreement",
    "LR-SVM disagreement",
    "LR-KNN disagreement",
    "RF-SVM disagreement",
    "RF-KNN disagreement",
    "SVM-KNN disagreement"
]

# Corresponding ROC-AUC values
auc_values = [
    lr_uncertainty_auc,
    rf_uncertainty_auc,
    svm_uncertainty_auc,
    knn_uncertainty_auc,
    lr_rf_auc,
    lr_svm_auc,
    lr_knn_auc,
    rf_svm_auc,
    rf_knn_auc,
    svm_knn_auc
]


# Create the bar graph
plt.figure(figsize=(12, 6))

plt.bar(labels, auc_values)

# Add a reference line at ROC-AUC = 0.50
plt.axhline(
    y=0.50,
    linestyle="--",
    linewidth=1
)

# Add labels and title
plt.xlabel("Uncertainty / Disagreement Measure")
plt.ylabel("ROC-AUC")
plt.title("ROC-AUC Comparison: Model Uncertainty vs Model Disagreement")

# Keep ROC-AUC between 0 and 1
plt.ylim(0, 1)

# Rotate labels so they are easier to read
plt.xticks(rotation=45, ha="right")

# Add the exact ROC-AUC value above each bar
for i, value in enumerate(auc_values):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.show()

# ============================================
# UNCERTAINTY VS DISAGREEMENT CORRELATION
# ============================================

# We will use Pearson correlation.
# It measures how strongly two numerical variables
# move together.
#
# +1 = strong positive relationship
#  0 = no linear relationship
# -1 = strong negative relationship


# ============================================
# 1. CALCULATE AVERAGE UNCERTAINTY FOR EACH PAIR
# ============================================

# Logistic Regression + Random Forest
disagreement_data["LR-RF Avg Uncertainty"] = (
    disagreement_data["LR Uncertainty"] +
    disagreement_data["RF Uncertainty"]
) / 2

# Logistic Regression + SVM
disagreement_data["LR-SVM Avg Uncertainty"] = (
    disagreement_data["LR Uncertainty"] +
    disagreement_data["SVM Uncertainty"]
) / 2

# Logistic Regression + KNN
disagreement_data["LR-KNN Avg Uncertainty"] = (
    disagreement_data["LR Uncertainty"] +
    disagreement_data["KNN Uncertainty"]
) / 2

# Random Forest + SVM
disagreement_data["RF-SVM Avg Uncertainty"] = (
    disagreement_data["RF Uncertainty"] +
    disagreement_data["SVM Uncertainty"]
) / 2

# Random Forest + KNN
disagreement_data["RF-KNN Avg Uncertainty"] = (
    disagreement_data["RF Uncertainty"] +
    disagreement_data["KNN Uncertainty"]
) / 2

# SVM + KNN
disagreement_data["SVM-KNN Avg Uncertainty"] = (
    disagreement_data["SVM Uncertainty"] +
    disagreement_data["KNN Uncertainty"]
) / 2


# ============================================
# 2. CALCULATE CORRELATIONS
# ============================================

correlations = {
    "LR vs RF": disagreement_data["LR-RF Avg Uncertainty"].corr(
        disagreement_data["LR vs RF"]
    ),

    "LR vs SVM": disagreement_data["LR-SVM Avg Uncertainty"].corr(
        disagreement_data["LR vs SVM"]
    ),

    "LR vs KNN": disagreement_data["LR-KNN Avg Uncertainty"].corr(
        disagreement_data["LR vs KNN"]
    ),

    "RF vs SVM": disagreement_data["RF-SVM Avg Uncertainty"].corr(
        disagreement_data["RF vs SVM"]
    ),

    "RF vs KNN": disagreement_data["RF-KNN Avg Uncertainty"].corr(
        disagreement_data["RF vs KNN"]
    ),

    "SVM vs KNN": disagreement_data["SVM-KNN Avg Uncertainty"].corr(
        disagreement_data["SVM vs KNN"]
    )
}


# ============================================
# 3. CREATE CORRELATION TABLE
# ============================================

correlation_table = pd.DataFrame(
    list(correlations.items()),
    columns=["Model Pair", "Pearson Correlation"]
)

print("UNCERTAINTY VS DISAGREEMENT CORRELATIONS")
print("-" * 50)

display(correlation_table.round(4))


# ============================================
# 4. SCATTER PLOTS
# ============================================

plot_data = [
    ("LR vs RF", "LR-RF Avg Uncertainty"),
    ("LR vs SVM", "LR-SVM Avg Uncertainty"),
    ("LR vs KNN", "LR-KNN Avg Uncertainty"),
    ("RF vs SVM", "RF-SVM Avg Uncertainty"),
    ("RF vs KNN", "RF-KNN Avg Uncertainty"),
    ("SVM vs KNN", "SVM-KNN Avg Uncertainty")
]


for pair_name, uncertainty_column in plot_data:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        disagreement_data[uncertainty_column],
        disagreement_data[pair_name],
        alpha=0.7
    )

    plt.xlabel("Average Model Uncertainty")
    plt.ylabel("Model Disagreement")
    plt.title(
        f"{pair_name}: Average Uncertainty vs Disagreement"
    )

    plt.grid(True, alpha=0.3)

    plt.show()


# ============================================
# 5. PRINT THE CORRELATION VALUES AGAIN
# ============================================

print("\nSUMMARY")
print("-" * 50)

for pair, correlation in correlations.items():
    print(f"{pair}: {correlation:.4f}")

# ============================================
# OUT-OF-FOLD ANALYSIS:
# DOES DISAGREEMENT ADD INFORMATION?
# ============================================

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ============================================
# 1. SET UP MODELS
# ============================================

models = {
    "LR": make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=5000)
    ),

    "RF": RandomForestClassifier(
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=42)
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
}

# ============================================
# 2. CREATE 5-FOLD CROSS-VALIDATION
# ============================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ============================================
# 3. GET OUT-OF-FOLD PROBABILITIES
# ============================================

oof_probabilities = {}

for name, model in models.items():

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    oof_probabilities[name] = probabilities

# ============================================
# 4. CREATE OUT-OF-FOLD PREDICTIONS
# ============================================

oof_predictions = {}

for name in models:

    oof_predictions[name] = (
        oof_probabilities[name] >= 0.5
    ).astype(int)

# ============================================
# 5. CREATE ERROR FLAGS
# ============================================

oof_errors = {}

for name in models:

    oof_errors[name] = (
        oof_predictions[name] != y.to_numpy()
    ).astype(int)

# ============================================
# 6. CALCULATE INDIVIDUAL UNCERTAINTY
# ============================================

oof_uncertainty = {}

for name in models:

    probability = oof_probabilities[name]

    uncertainty = (
        1 - 2 * np.abs(probability - 0.5)
    )

    oof_uncertainty[name] = uncertainty

# ============================================
# 7. CALCULATE MODEL DISAGREEMENT
# ============================================

model_pairs = [
    ("LR", "RF"),
    ("LR", "SVM"),
    ("LR", "KNN"),
    ("RF", "SVM"),
    ("RF", "KNN"),
    ("SVM", "KNN")
]

oof_disagreement = {}

for model_1, model_2 in model_pairs:

    pair_name = f"{model_1} vs {model_2}"

    disagreement = np.abs(
        oof_probabilities[model_1]
        -
        oof_probabilities[model_2]
    )

    oof_disagreement[pair_name] = disagreement

# ============================================
# 8. DISPLAY HOW MANY ERRORS EACH MODEL MADE
# ============================================

error_summary = pd.DataFrame({
    "Model": list(models.keys()),
    "Total Errors": [
        oof_errors[name].sum()
        for name in models
    ],
    "Error Rate": [
        oof_errors[name].mean()
        for name in models
    ]
})

error_summary["Error Rate"] = (
    error_summary["Error Rate"].round(4)
)

print("OUT-OF-FOLD ERROR SUMMARY")
display(error_summary)

# ============================================
# 9. TEST UNCERTAINTY VS DISAGREEMENT
# ============================================

results = []

for model_1, model_2 in model_pairs:

    pair_name = f"{model_1} vs {model_2}"

    disagreement = oof_disagreement[pair_name]

    # Test disagreement against BOTH models' errors
    for target_model in [model_1, model_2]:

        error = oof_errors[target_model]

        uncertainty = oof_uncertainty[target_model]

        # ------------------------------------
        # Individual uncertainty
        # ------------------------------------

        uncertainty_auc = roc_auc_score(
            error,
            uncertainty
        )

        # ------------------------------------
        # Disagreement
        # ------------------------------------

        disagreement_auc = roc_auc_score(
            error,
            disagreement
        )

        # ------------------------------------
        # Combined model
        # ------------------------------------
        #
        # Input 1 = target model uncertainty
        # Input 2 = model disagreement
        #
        # This directly tests whether
        # disagreement adds information
        # beyond individual uncertainty.
        # ------------------------------------

        combined_features = np.column_stack([
            uncertainty,
            disagreement
        ])

        combined_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                random_state=42,
                max_iter=5000
            )
        )

        combined_probability = cross_val_predict(
            combined_model,
            combined_features,
            error,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        combined_auc = roc_auc_score(
            error,
            combined_probability
        )

        # ------------------------------------
        # Improvement from adding disagreement
        # ------------------------------------

        improvement = (
            combined_auc - uncertainty_auc
        )

        results.append({
            "Model Pair": pair_name,
            "Target Model": target_model,
            "Uncertainty ROC-AUC": uncertainty_auc,
            "Disagreement ROC-AUC": disagreement_auc,
            "Both Together ROC-AUC": combined_auc,
            "Improvement From Disagreement": improvement
        })

# ============================================
# 10. CREATE RESULTS TABLE
# ============================================

comparison_table = pd.DataFrame(results)

numeric_columns = [
    "Uncertainty ROC-AUC",
    "Disagreement ROC-AUC",
    "Both Together ROC-AUC",
    "Improvement From Disagreement"
]

comparison_table[numeric_columns] = (
    comparison_table[numeric_columns].round(4)
)

print("DOES DISAGREEMENT ADD INFORMATION BEYOND UNCERTAINTY?")

display(comparison_table)

# ============================================
# 11. OVERALL RESULTS
# ============================================

print("\nAVERAGE PERFORMANCE")

average_results = pd.DataFrame({
    "Method": [
        "Individual Uncertainty",
        "Disagreement",
        "Both Together"
    ],
    "Mean ROC-AUC": [
        comparison_table["Uncertainty ROC-AUC"].mean(),
        comparison_table["Disagreement ROC-AUC"].mean(),
        comparison_table["Both Together ROC-AUC"].mean()
    ]
})

average_results["Mean ROC-AUC"] = (
    average_results["Mean ROC-AUC"].round(4)
)

display(average_results)

# ============================================
# 12. AVERAGE IMPROVEMENT
# ============================================

mean_improvement = (
    comparison_table["Improvement From Disagreement"].mean()
)

print(
    f"\nAverage improvement from adding disagreement: "
    f"{mean_improvement:.4f}"
)

# ============================================================
# RISK SCORE ANALYSIS
# UNCERTAINTY + MODEL DISAGREEMENT
# ============================================================

# This section uses the OUT-OF-FOLD values already calculated
# above.
#
# We are NOT training another machine-learning model.
#
# We are creating three simple, interpretable risk scores:
#
# Risk Score A = Average Model Uncertainty
# Risk Score B = Average Model Disagreement
# Risk Score C = Combined Standardized Uncertainty + Disagreement
#
# There are 569 observations in the Wisconsin Breast Cancer dataset.


# ============================================================
# 1. CREATE A NEW RISK ANALYSIS DATAFRAME
# ============================================================

risk_data = pd.DataFrame({
    "Observation": range(len(y)),

    # Actual diagnosis
    "Actual": y.to_numpy(),

    # --------------------------------------------------------
    # Individual model uncertainties
    # --------------------------------------------------------
    "LR Uncertainty": oof_uncertainty["LR"],
    "RF Uncertainty": oof_uncertainty["RF"],
    "SVM Uncertainty": oof_uncertainty["SVM"],
    "KNN Uncertainty": oof_uncertainty["KNN"],

    # --------------------------------------------------------
    # Six pairwise disagreements
    # --------------------------------------------------------
    "LR vs RF": oof_disagreement["LR vs RF"],
    "LR vs SVM": oof_disagreement["LR vs SVM"],
    "LR vs KNN": oof_disagreement["LR vs KNN"],
    "RF vs SVM": oof_disagreement["RF vs SVM"],
    "RF vs KNN": oof_disagreement["RF vs KNN"],
    "SVM vs KNN": oof_disagreement["SVM vs KNN"],

    # --------------------------------------------------------
    # Whether each model was wrong
    # --------------------------------------------------------
    "LR Wrong": oof_errors["LR"],
    "RF Wrong": oof_errors["RF"],
    "SVM Wrong": oof_errors["SVM"],
    "KNN Wrong": oof_errors["KNN"]
})


# ============================================================
# 2. CALCULATE AVERAGE MODEL UNCERTAINTY
# ============================================================

uncertainty_columns = [
    "LR Uncertainty",
    "RF Uncertainty",
    "SVM Uncertainty",
    "KNN Uncertainty"
]

risk_data["Average Uncertainty"] = (
    risk_data[uncertainty_columns].mean(axis=1)
)


# ============================================================
# 3. CALCULATE AVERAGE PAIRWISE DISAGREEMENT
# ============================================================

disagreement_columns = [
    "LR vs RF",
    "LR vs SVM",
    "LR vs KNN",
    "RF vs SVM",
    "RF vs KNN",
    "SVM vs KNN"
]

risk_data["Average Disagreement"] = (
    risk_data[disagreement_columns].mean(axis=1)
)


# ============================================================
# 4. CALCULATE MAXIMUM PAIRWISE DISAGREEMENT
# ============================================================

risk_data["Maximum Disagreement"] = (
    risk_data[disagreement_columns].max(axis=1)
)


# ============================================================
# 5. CALCULATE NUMBER OF MODELS THAT WERE WRONG
# ============================================================

wrong_columns = [
    "LR Wrong",
    "RF Wrong",
    "SVM Wrong",
    "KNN Wrong"
]

risk_data["Number of Models Wrong"] = (
    risk_data[wrong_columns].sum(axis=1)
)


# ============================================================
# 6. CALCULATE WHETHER AT LEAST ONE MODEL WAS WRONG
# ============================================================

risk_data["At Least One Wrong"] = (
    risk_data["Number of Models Wrong"] > 0
).astype(int)


# ============================================================
# 7. CREATE RISK SCORE A
# ============================================================

# Risk Score A is simply the average uncertainty
# of the four models.

risk_data["Risk Score A"] = (
    risk_data["Average Uncertainty"]
)


# ============================================================
# 8. CREATE RISK SCORE B
# ============================================================

# Risk Score B is the average disagreement
# across all six model pairs.

risk_data["Risk Score B"] = (
    risk_data["Average Disagreement"]
)


# ============================================================
# 9. CREATE RISK SCORE C
# ============================================================

# Risk Score C combines:
#
# 1. Average uncertainty
# 2. Average disagreement
#
# These two values are standardized first so they are
# placed on comparable scales.
#
# Then they are combined equally.


# Standardize average uncertainty
uncertainty_mean = risk_data["Average Uncertainty"].mean()
uncertainty_std = risk_data["Average Uncertainty"].std()

risk_data["Standardized Uncertainty"] = (
    (risk_data["Average Uncertainty"] - uncertainty_mean)
    / uncertainty_std
)


# Standardize average disagreement
disagreement_mean = risk_data["Average Disagreement"].mean()
disagreement_std = risk_data["Average Disagreement"].std()

risk_data["Standardized Disagreement"] = (
    (risk_data["Average Disagreement"] - disagreement_mean)
    / disagreement_std
)


# Combine both standardized values equally
risk_data["Risk Score C"] = (
    risk_data["Standardized Uncertainty"]
    + risk_data["Standardized Disagreement"]
) / 2


# ============================================================
# 10. CHECK THE RISK DATA
# ============================================================

print("RISK ANALYSIS DATA")
print("-" * 60)

print("Number of observations:", len(risk_data))

print("\nRisk score columns:")
print([
    "Risk Score A",
    "Risk Score B",
    "Risk Score C"
])

display(
    risk_data[
        [
            "Observation",
            "Average Uncertainty",
            "Average Disagreement",
            "Maximum Disagreement",
            "Number of Models Wrong",
            "At Least One Wrong",
            "Risk Score A",
            "Risk Score B",
            "Risk Score C"
        ]
    ].head(10).round(4)
)


# ============================================================
# DEBUGGED RISK-ANALYSIS REPORTING SECTION
# ============================================================
#
# IMPORTANT:
# This section does NOT retrain any models.
# It does NOT change the risk-score calculations.
# It only rebuilds the reporting calculations from:
#
#   oof_errors["LR"]
#   oof_errors["RF"]
#   oof_errors["SVM"]
#   oof_errors["KNN"]
#
# and the already-created:
#
#   risk_data["Risk Score A"]
#   risk_data["Risk Score B"]
#
# ============================================================


# ============================================================
# 1. REBUILD NUMBER OF MODELS WRONG DIRECTLY FROM OOF ERRORS
# ============================================================

# Each OOF error array contains:
# 0 = model was correct
# 1 = model was wrong
#
# Adding the four arrays gives the number of models
# that were wrong for each individual observation.

risk_data["Number of Models Wrong"] = (
    np.asarray(oof_errors["LR"], dtype=int)
    + np.asarray(oof_errors["RF"], dtype=int)
    + np.asarray(oof_errors["SVM"], dtype=int)
    + np.asarray(oof_errors["KNN"], dtype=int)
)


# ============================================================
# 2. REBUILD "AT LEAST ONE WRONG" DIRECTLY FROM THE FOUR
#    OOF ERROR ARRAYS
# ============================================================

risk_data["At Least One Wrong"] = (
    risk_data["Number of Models Wrong"] > 0
).astype(int)


# ============================================================
# 3. SANITY CHECK THE ERROR COUNTS
# ============================================================

print("ERROR DATA CHECK")
print("-" * 60)

print(
    "Total observations:",
    len(risk_data)
)

print(
    "Total observations with at least one model wrong:",
    int(risk_data["At Least One Wrong"].sum())
)

print(
    "Overall at-least-one-model error rate:",
    f"{risk_data['At Least One Wrong'].mean():.4f}",
    f"({risk_data['At Least One Wrong'].mean() * 100:.2f}%)"
)

print(
    "\nMinimum number of models wrong:",
    risk_data["Number of Models Wrong"].min()
)

print(
    "Maximum number of models wrong:",
    risk_data["Number of Models Wrong"].max()
)

# This MUST be between 0 and 4.
assert risk_data["Number of Models Wrong"].between(0, 4).all(), (
    "ERROR: Number of Models Wrong contains a value outside 0-4."
)


# ============================================================
# 4. VERIFY RISK SCORE A AND RISK SCORE B
# ============================================================

# Risk Score A should equal average uncertainty.
assert np.allclose(
    risk_data["Risk Score A"],
    risk_data["Average Uncertainty"]
), "Risk Score A does not match Average Uncertainty."

# Risk Score B should equal average disagreement.
assert np.allclose(
    risk_data["Risk Score B"],
    risk_data["Average Disagreement"]
), "Risk Score B does not match Average Disagreement."

print("\nRisk Score A verified:")
print("Risk Score A = Average Model Uncertainty")

print("\nRisk Score B verified:")
print("Risk Score B = Average Model Disagreement")


# ============================================================
# 5. RECONSTRUCT RISK SCORE C FROM STANDARDIZED A AND B
# ============================================================

# Standardize Risk Score A
risk_a_mean = risk_data["Risk Score A"].mean()
risk_a_std = risk_data["Risk Score A"].std()

standardized_risk_a = (
    risk_data["Risk Score A"] - risk_a_mean
) / risk_a_std


# Standardize Risk Score B
risk_b_mean = risk_data["Risk Score B"].mean()
risk_b_std = risk_data["Risk Score B"].std()

standardized_risk_b = (
    risk_data["Risk Score B"] - risk_b_mean
) / risk_b_std


# Combine them equally
risk_c_check = (
    standardized_risk_a +
    standardized_risk_b
) / 2


# ============================================================
# 6. VERIFY THE EXISTING RISK SCORE C
# ============================================================

risk_c_matches = np.allclose(
    risk_data["Risk Score C"],
    risk_c_check
)

print("\nRisk Score C verification:")
print("Risk Score C = standardized Risk Score A +")
print("                standardized Risk Score B")
print("              --------------------------------")
print("                         2")

print(
    "\nRisk Score C matches this calculation:",
    risk_c_matches
)

assert risk_c_matches, (
    "ERROR: Existing Risk Score C does not match "
    "the standardized Risk Score A/B calculation."
)


# ============================================================
# 7. CHECK THAT HIGHER RISK SCORE C MEANS HIGHER COMBINED
#    STANDARDIZED RISK
# ============================================================

# Because Risk Score C is a numerical combination of the two
# standardized components, sorting Risk Score C from low to
# high should produce increasing combined-score values.

risk_c_sorted = risk_data["Risk Score C"].sort_values()

is_monotonic = risk_c_sorted.is_monotonic_increasing

print(
    "\nRisk Score C increases from low to high:",
    is_monotonic
)

assert is_monotonic, (
    "ERROR: Risk Score C is not ordered correctly."
)


# ============================================================
# 8. FUNCTION TO CREATE EXACT QUARTILES
# ============================================================

def make_risk_quartiles(data, score_column):

    # Rank first so repeated risk scores do not cause
    # qcut() to create invalid bins.
    ranked_scores = data[score_column].rank(
        method="first"
    )

    return pd.qcut(
        ranked_scores,
        q=4,
        labels=[
            "Lowest 25%",
            "25-50%",
            "50-75%",
            "Highest 25%"
        ]
    )


# ============================================================
# 9. FUNCTION TO CALCULATE CORRECT QUARTILE STATISTICS
# ============================================================

def calculate_correct_quartile_table(data, score_column):

    temp = data.copy()

    temp["Risk Quartile"] = make_risk_quartiles(
        temp,
        score_column
    )

    quartile_order = [
        "Lowest 25%",
        "25-50%",
        "50-75%",
        "Highest 25%"
    ]

    results = []

    for quartile in quartile_order:

        group = temp[
            temp["Risk Quartile"] == quartile
        ]

        number_observations = len(group)

        # Number of observations with at least one error
        number_with_error = int(
            group["At Least One Wrong"].sum()
        )

        # Error rate as a DECIMAL
        error_rate_decimal = (
            number_with_error /
            number_observations
        )

        # Error rate as a PERCENTAGE
        error_rate_percent = (
            error_rate_decimal * 100
        )

        # Average number of models wrong.
        #
        # IMPORTANT:
        # This is calculated directly from the number of
        # models wrong per observation.
        average_models_wrong = (
            group["Number of Models Wrong"].mean()
        )

        results.append({
            "Risk Score": score_column,
            "Risk Quartile": quartile,
            "Observations": number_observations,
            "Observations At Least One Wrong": number_with_error,
            "Error Rate (Decimal)": error_rate_decimal,
            "Error Rate (%)": error_rate_percent,
            "Average Models Wrong": average_models_wrong
        })

    return pd.DataFrame(results)


# ============================================================
# 10. CREATE QUARTILE TABLES FOR ALL THREE RISK SCORES
# ============================================================

risk_score_columns = [
    "Risk Score A",
    "Risk Score B",
    "Risk Score C"
]

quartile_tables = []

for score_column in risk_score_columns:

    table = calculate_correct_quartile_table(
        risk_data,
        score_column
    )

    quartile_tables.append(table)


quartile_comparison = pd.concat(
    quartile_tables,
    ignore_index=True
)


# ============================================================
# 11. CHECK QUARTILE COUNTS
# ============================================================

print("\nQUARTILE COUNT CHECK")
print("-" * 60)

for score_column in risk_score_columns:

    score_counts = (
        quartile_comparison[
            quartile_comparison["Risk Score"] ==
            score_column
        ]["Observations"]
    )

    print(
        f"{score_column}:",
        score_counts.tolist(),
        "Total =",
        score_counts.sum()
    )

    assert score_counts.sum() == len(risk_data), (
        f"ERROR: {score_column} quartiles do not "
        "contain all observations."
    )


# ============================================================
# 12. CHECK THAT AVERAGE MODELS WRONG IS BETWEEN 0 AND 4
# ============================================================

assert (
    quartile_comparison["Average Models Wrong"]
    .between(0, 4)
    .all()
)

print(
    "\n✓ Every Average Models Wrong value is between 0 and 4."
)


# ============================================================
# 13. CHECK THAT ERROR RATE IS CALCULATED CORRECTLY
# ============================================================

calculated_error_rate = (
    quartile_comparison[
        "Observations At Least One Wrong"
    ]
    /
    quartile_comparison["Observations"]
)

assert np.allclose(
    quartile_comparison["Error Rate (Decimal)"],
    calculated_error_rate
)

print(
    "✓ Every quartile Error Rate is calculated as "
    "errors / observations."
)


# ============================================================
# 14. DISPLAY CORRECTED QUARTILE COMPARISON TABLE
# ============================================================

print("\nCORRECTED RISK SCORE QUARTILE COMPARISON")
print("-" * 80)

display(
    quartile_comparison[
        [
            "Risk Score",
            "Risk Quartile",
            "Observations",
            "Observations At Least One Wrong",
            "Error Rate (Decimal)",
            "Error Rate (%)",
            "Average Models Wrong"
        ]
    ].round(4)
)


# ============================================================
# 15. CALCULATE FINAL SUMMARY
# ============================================================

total_at_least_one_wrong = int(
    risk_data["At Least One Wrong"].sum()
)

final_summary = []


for score_column in risk_score_columns:

    # Sort from lowest risk to highest risk
    sorted_data = risk_data.sort_values(
        by=score_column,
        ascending=True
    )

    # Highest 25%.
    #
    # 569 * 0.25 = 142.25
    #
    # The quartile procedure above produces:
    # 143, 142, 142, 142
    #
    # Therefore the highest quartile contains 142
    # observations.
    highest_risk_quartile = (
        make_risk_quartiles(
            risk_data,
            score_column
        )
        == "Highest 25%"
    )

    highest_risk_data = risk_data[
        highest_risk_quartile
    ]

    highest_risk_observations = len(
        highest_risk_data
    )

    highest_risk_errors = int(
        highest_risk_data["At Least One Wrong"].sum()
    )

    # Error rate as decimal
    highest_risk_error_rate_decimal = (
        highest_risk_errors /
        highest_risk_observations
    )

    # Error rate as percentage
    highest_risk_error_rate_percent = (
        highest_risk_error_rate_decimal * 100
    )

    # Percentage of ALL observations with at least one
    # model error that occur in the highest-risk quartile.
    #
    # Denominator = ALL observations with at least one
    # model error.
    errors_captured_percent = (
        highest_risk_errors /
        total_at_least_one_wrong
    ) * 100

    # ROC-AUC
    auc = roc_auc_score(
        risk_data["At Least One Wrong"],
        risk_data[score_column]
    )

    final_summary.append({
        "Risk Score": score_column,
        "ROC-AUC": auc,
        "Highest-Risk-25% Error Rate (Decimal)": (
            highest_risk_error_rate_decimal
        ),
        "Highest-Risk-25% Error Rate (%)": (
            highest_risk_error_rate_percent
        ),
        "% of All Errors Captured in Highest-Risk 25%": (
            errors_captured_percent
        ),
        "Highest-Risk-25% Observations": (
            highest_risk_observations
        ),
        "Errors in Highest-Risk-25%": (
            highest_risk_errors
        )
    })


final_risk_summary = pd.DataFrame(
    final_summary
)


# ============================================================
# 16. VERIFY FINAL SUMMARY AGAINST QUARTILE TABLE
# ============================================================

for score_column in risk_score_columns:

    # Get highest-risk row from quartile table
    quartile_row = quartile_comparison[
        (quartile_comparison["Risk Score"] == score_column)
        &
        (quartile_comparison["Risk Quartile"] == "Highest 25%")
    ].iloc[0]

    # Get corresponding final-summary row
    summary_row = final_risk_summary[
        final_risk_summary["Risk Score"] == score_column
    ].iloc[0]

    # The two error rates MUST match
    assert np.isclose(
        quartile_row["Error Rate (%)"],
        summary_row["Highest-Risk-25% Error Rate (%)"]
    ), (
        f"ERROR: Highest-risk error rate does not match "
        f"for {score_column}."
    )


print(
    "\n✓ Highest-risk quartile error rates exactly match "
    "the final summary."
)


# ============================================================
# 17. DISPLAY CORRECTED FINAL SUMMARY TABLE
# ============================================================

print("\nCORRECTED FINAL RISK SCORE SUMMARY")
print("-" * 90)

display(
    final_risk_summary[
        [
            "Risk Score",
            "ROC-AUC",
            "Highest-Risk-25% Error Rate (Decimal)",
            "Highest-Risk-25% Error Rate (%)",
            "% of All Errors Captured in Highest-Risk 25%"
        ]
    ].round(4)
)


# ============================================================
# 18. FINAL DEBUG CHECKS
# ============================================================

print("\nFINAL DEBUG CHECKS")
print("-" * 60)

print(
    "Total observations:",
    len(risk_data)
)

print(
    "Total observations with at least one model wrong:",
    total_at_least_one_wrong
)

print(
    "Overall error rate:",
    f"{risk_data['At Least One Wrong'].mean():.4f}",
    f"({risk_data['At Least One Wrong'].mean() * 100:.2f}%)"
)

print(
    "\nNumber of models wrong range:",
    risk_data["Number of Models Wrong"].min(),
    "to",
    risk_data["Number of Models Wrong"].max()
)

print(
    "\nRisk Score C calculation verified:",
    risk_c_matches
)

print(
    "Risk Score C sorted correctly:",
    is_monotonic
)

print(
    "\n✓ Risk-analysis reporting section completed."
)

# ============================================================
# AI RELIABILITY MONITOR
# RESEARCH PROTOTYPE
# ============================================================
#
# IMPORTANT:
# This is a research prototype demonstrating the concept.
# It is NOT a clinical diagnostic tool.
#
# The prototype uses the OUT-OF-FOLD model probabilities
# and risk-analysis values already calculated above.
#
# It does NOT retrain the models.
# It does NOT modify the previous scientific analysis.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import sys
import json
import pickle
import shutil
import subprocess
import importlib.util

import numpy as np
import pandas as pd


# ============================================================
# 2. VERIFY THAT THE REQUIRED OOF DATA EXISTS
# ============================================================

required_objects = [
    "risk_data",
    "oof_probabilities",
    "oof_uncertainty",
    "oof_disagreement"
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        "The following required analysis objects are missing: "
        + ", ".join(missing_objects)
        + "\nRun the previous OOF/risk-analysis sections first."
    )

print("AI Reliability Monitor data check passed.")


# ============================================================
# 3. CREATE PROTOTYPE DATA FROM EXISTING OOF RESULTS
# ============================================================

prototype_data = risk_data.copy()

# Store the OOF benign probabilities.
#
# These are the probabilities already calculated during
# the out-of-fold analysis.

prototype_data["LR Probability"] = oof_probabilities["LR"]
prototype_data["RF Probability"] = oof_probabilities["RF"]
prototype_data["SVM Probability"] = oof_probabilities["SVM"]
prototype_data["KNN Probability"] = oof_probabilities["KNN"]


# ============================================================
# 4. CALCULATE OOF PREDICTED CLASSES
# ============================================================

prototype_data["LR Prediction"] = (
    prototype_data["LR Probability"] >= 0.5
).astype(int)

prototype_data["RF Prediction"] = (
    prototype_data["RF Probability"] >= 0.5
).astype(int)

prototype_data["SVM Prediction"] = (
    prototype_data["SVM Probability"] >= 0.5
).astype(int)

prototype_data["KNN Prediction"] = (
    prototype_data["KNN Probability"] >= 0.5
).astype(int)


# ============================================================
# 5. VERIFY UNCERTAINTY VALUES
# ============================================================

# Use the same uncertainty formula from the research:
#
# Uncertainty = 1 - 2 * abs(P - 0.5)

prototype_data["LR Uncertainty"] = (
    1 - 2 * np.abs(prototype_data["LR Probability"] - 0.5)
)

prototype_data["RF Uncertainty"] = (
    1 - 2 * np.abs(prototype_data["RF Probability"] - 0.5)
)

prototype_data["SVM Uncertainty"] = (
    1 - 2 * np.abs(prototype_data["SVM Probability"] - 0.5)
)

prototype_data["KNN Uncertainty"] = (
    1 - 2 * np.abs(prototype_data["KNN Probability"] - 0.5)
)


# ============================================================
# 6. VERIFY THE AVERAGE UNCERTAINTY
# ============================================================

prototype_data["Average Uncertainty"] = (
    prototype_data[
        [
            "LR Uncertainty",
            "RF Uncertainty",
            "SVM Uncertainty",
            "KNN Uncertainty"
        ]
    ].mean(axis=1)
)


# ============================================================
# 7. CALCULATE ALL SIX PAIRWISE DISAGREEMENTS
# ============================================================

prototype_data["LR vs RF"] = np.abs(
    prototype_data["LR Probability"]
    -
    prototype_data["RF Probability"]
)

prototype_data["LR vs SVM"] = np.abs(
    prototype_data["LR Probability"]
    -
    prototype_data["SVM Probability"]
)

prototype_data["LR vs KNN"] = np.abs(
    prototype_data["LR Probability"]
    -
    prototype_data["KNN Probability"]
)

prototype_data["RF vs SVM"] = np.abs(
    prototype_data["RF Probability"]
    -
    prototype_data["SVM Probability"]
)

prototype_data["RF vs KNN"] = np.abs(
    prototype_data["RF Probability"]
    -
    prototype_data["KNN Probability"]
)

prototype_data["SVM vs KNN"] = np.abs(
    prototype_data["SVM Probability"]
    -
    prototype_data["KNN Probability"]
)


pair_columns = [
    "LR vs RF",
    "LR vs SVM",
    "LR vs KNN",
    "RF vs SVM",
    "RF vs KNN",
    "SVM vs KNN"
]


# ============================================================
# 8. CALCULATE AVERAGE AND MAXIMUM DISAGREEMENT
# ============================================================

prototype_data["Average Disagreement"] = (
    prototype_data[pair_columns].mean(axis=1)
)

prototype_data["Maximum Disagreement"] = (
    prototype_data[pair_columns].max(axis=1)
)


# ============================================================
# 9. VERIFY THE RISK SCORE C STANDARDIZATION
# ============================================================

# IMPORTANT:
# We use the SAME mean and standard deviation from the
# complete 569-observation risk-analysis dataset.

uncertainty_mean = risk_data["Average Uncertainty"].mean()
uncertainty_std = risk_data["Average Uncertainty"].std()

disagreement_mean = risk_data["Average Disagreement"].mean()
disagreement_std = risk_data["Average Disagreement"].std()


prototype_data["Standardized Uncertainty"] = (
    (
        prototype_data["Average Uncertainty"]
        -
        uncertainty_mean
    )
    /
    uncertainty_std
)

prototype_data["Standardized Disagreement"] = (
    (
        prototype_data["Average Disagreement"]
        -
        disagreement_mean
    )
    /
    disagreement_std
)


# Same validated Risk Score C formula
prototype_data["Risk Score C"] = (
    prototype_data["Standardized Uncertainty"]
    +
    prototype_data["Standardized Disagreement"]
) / 2


# ============================================================
# 10. VERIFY RISK SCORE C AGAINST THE EXISTING RISK DATA
# ============================================================

risk_score_difference = np.max(
    np.abs(
        prototype_data["Risk Score C"].to_numpy()
        -
        risk_data["Risk Score C"].to_numpy()
    )
)

print(
    "Maximum difference between existing and prototype "
    f"Risk Score C: {risk_score_difference:.12f}"
)

if risk_score_difference < 1e-10:
    print("✓ Risk Score C matches the existing risk analysis.")
else:
    print(
        "WARNING: Prototype Risk Score C does not exactly "
        "match the existing risk analysis."
    )


# ============================================================
# 11. CREATE RISK QUARTILES
# ============================================================

# Rank first so repeated scores do not cause qcut problems.

ranked_risk = prototype_data["Risk Score C"].rank(
    method="first"
)

prototype_data["Risk Quartile"] = pd.qcut(
    ranked_risk,
    q=4,
    labels=[
        "Lowest 25%",
        "25–50%",
        "50–75%",
        "Highest 25%"
    ]
)


# ============================================================
# 12. CREATE RELIABILITY FLAGS
# ============================================================

reliability_flags = {
    "Lowest 25%": "LOW RISK",
    "25–50%": "MODERATE-LOW RISK",
    "50–75%": "MODERATE-HIGH RISK",
    "Highest 25%": "HIGH RISK / HUMAN REVIEW"
}

prototype_data["Reliability Flag"] = (
    prototype_data["Risk Quartile"]
    .astype(str)
    .map(reliability_flags)
)


# ============================================================
# 13. VERIFY QUARTILE COUNTS
# ============================================================

print("\nPROTOTYPE QUARTILE COUNTS")
print("-" * 50)

print(
    prototype_data["Risk Quartile"].value_counts(
        sort=False
    )
)


# ============================================================
# 14. SAVE PROTOTYPE DATA
# ============================================================

prototype_data_path = "/content/ai_reliability_monitor_data.pkl"

with open(prototype_data_path, "wb") as file:
    pickle.dump(
        {
            "prototype_data": prototype_data,
            "uncertainty_mean": uncertainty_mean,
            "uncertainty_std": uncertainty_std,
            "disagreement_mean": disagreement_mean,
            "disagreement_std": disagreement_std
        },
        file
    )

print(
    "\nPrototype data prepared successfully:"
)
print(prototype_data_path)


# ============================================================
# 15. CHECK WHETHER STREAMLIT IS AVAILABLE
# ============================================================

streamlit_available = (
    importlib.util.find_spec("streamlit") is not None
)

print(
    "\nStreamlit available:",
    streamlit_available
)


# ============================================================
# 16. CREATE THE STREAMLIT APPLICATION
# ============================================================

streamlit_app = r'''
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Reliability Monitor",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

with open("ai_reliability_monitor_data.pkl", "rb") as file:
    saved_data = pickle.load(file)

data = saved_data["prototype_data"]


# ============================================================
# HEADER
# ============================================================

st.title("AI Reliability Monitor")

st.caption(
    "Research Prototype — Model Reliability Monitoring"
)

st.warning(
    "This is a research prototype demonstrating a model "
    "reliability concept. It is NOT a clinical diagnostic tool."
)


st.markdown(
    """
This monitor examines uncertainty and disagreement among
four independently trained machine-learning models.

It does **not** determine whether a prediction is correct.
It estimates how much additional review may be warranted
based on model uncertainty and disagreement.
"""
)


# ============================================================
# SELECT OBSERVATION
# ============================================================

st.subheader("Select Test Observation")

observation_options = data["Observation"].tolist()

selected_observation = st.selectbox(
    "Observation",
    observation_options
)

row = data[
    data["Observation"] == selected_observation
].iloc[0]


# ============================================================
# MODEL INFORMATION
# ============================================================

models = [
    ("LR", "Logistic Regression"),
    ("RF", "Random Forest"),
    ("SVM", "SVM"),
    ("KNN", "KNN")
]

model_rows = []

for short_name, model_name in models:

    prediction = int(row[f"{short_name} Prediction"])
    probability = float(row[f"{short_name} Probability"])
    uncertainty = float(row[f"{short_name} Uncertainty"])

    prediction_label = (
        "Benign"
        if prediction == 1
        else "Malignant"
    )

    model_rows.append({
        "MODEL": model_name,
        "PREDICTION": prediction_label,
        "BENIGN PROBABILITY": probability,
        "UNCERTAINTY": uncertainty
    })

model_table = pd.DataFrame(model_rows)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("Model Comparison")

display_table = model_table.copy()

display_table["BENIGN PROBABILITY"] = (
    display_table["BENIGN PROBABILITY"]
    .map(lambda x: f"{x:.4f}")
)

display_table["UNCERTAINTY"] = (
    display_table["UNCERTAINTY"]
    .map(lambda x: f"{x:.4f}")
)

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CLASS AGREEMENT
# ============================================================

predictions = [
    row["LR Prediction"],
    row["RF Prediction"],
    row["SVM Prediction"],
    row["KNN Prediction"]
]

all_models_agree = len(set(predictions)) == 1

if all_models_agree:

    st.success(
        "✓ All four models agree on the predicted class."
    )

else:

    st.warning(
        "⚠ The four models disagree on the predicted class."
    )


# ============================================================
# RISK METRICS
# ============================================================

st.subheader("Model Reliability Risk")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Uncertainty",
        f"{row['Average Uncertainty']:.4f}"
    )

with col2:
    st.metric(
        "Average Disagreement",
        f"{row['Average Disagreement']:.4f}"
    )

with col3:
    st.metric(
        "Maximum Disagreement",
        f"{row['Maximum Disagreement']:.4f}"
    )


st.metric(
    "Combined Risk Score",
    f"{row['Risk Score C']:.4f}"
)


# ============================================================
# RISK QUARTILE AND FLAG
# ============================================================

quartile = str(row["Risk Quartile"])
flag = row["Reliability Flag"]


if quartile == "Highest 25%":

    st.error(
        f"⚠ {flag}"
    )

elif quartile == "50–75%":

    st.warning(
        f"{flag}"
    )

elif quartile == "25–50%":

    st.info(
        f"{flag}"
    )

else:

    st.success(
        f"{flag}"
    )


st.write(
    f"**Risk Quartile:** {quartile}"
)


# ============================================================
# SIX PAIRWISE DISAGREEMENTS
# ============================================================

st.subheader("Pairwise Model Disagreement")

pair_display = pd.DataFrame({
    "Model Pair": [
        "LR vs RF",
        "LR vs SVM",
        "LR vs KNN",
        "RF vs SVM",
        "RF vs KNN",
        "SVM vs KNN"
    ],
    "Disagreement": [
        row["LR vs RF"],
        row["LR vs SVM"],
        row["LR vs KNN"],
        row["RF vs SVM"],
        row["RF vs KNN"],
        row["SVM vs KNN"]
    ]
})

pair_display["Disagreement"] = (
    pair_display["Disagreement"]
    .map(lambda x: f"{x:.4f}")
)

st.dataframe(
    pair_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROBABILITY VISUALIZATION
# ============================================================

st.subheader("Model Benign Probability")

probability_names = [
    "LR",
    "RF",
    "SVM",
    "KNN"
]

probabilities = [
    row["LR Probability"],
    row["RF Probability"],
    row["SVM Probability"],
    row["KNN Probability"]
]

fig, ax = plt.subplots(figsize=(10, 3.5))

y_positions = np.arange(len(probability_names))

ax.hlines(
    y_positions,
    0,
    1,
    linewidth=2
)

ax.scatter(
    probabilities,
    y_positions,
    s=100
)

ax.axvline(
    0.5,
    linestyle="--",
    linewidth=1
)

ax.set_xlim(0, 1)
ax.set_yticks(y_positions)
ax.set_yticklabels(probability_names)

ax.set_xlabel(
    "Probability of Benign"
)

ax.set_title(
    "Four Model Probability Comparison"
)

ax.grid(
    axis="x",
    alpha=0.3
)

for y, probability in zip(
    y_positions,
    probabilities
):
    ax.text(
        probability,
        y + 0.18,
        f"{probability:.3f}",
        ha="center"
    )

plt.tight_layout()

st.pyplot(fig)


st.caption(
    "A value near 0.5 represents greater uncertainty under "
    "the uncertainty formula used in this research."
)


# ============================================================
# INTERPRETATION
# ============================================================

st.subheader("Interpretation")

st.info(
    "This monitor does not determine whether a prediction is "
    "correct. It estimates how much additional review may be "
    "warranted based on model uncertainty and disagreement."
)


# ============================================================
# PROTOTYPE NOTICE
# ============================================================

st.caption(
    "AI Reliability Monitor — Research Prototype. "
    "Not a clinical diagnostic tool."
)
'''


# ============================================================
# 17. WRITE THE STREAMLIT FILE
# ============================================================

app_path = "/content/ai_reliability_monitor.py"

with open(app_path, "w", encoding="utf-8") as file:
    file.write(streamlit_app)

print(
    "\nStreamlit application created:"
)
print(app_path)


# ============================================================
# 18. LAUNCH INSTRUCTIONS
# ============================================================

print("\n" + "=" * 70)
print("AI RELIABILITY MONITOR — LAUNCH INSTRUCTIONS")
print("=" * 70)

if streamlit_available:

    print("""
Streamlit is available in this environment.

The application file has been created:

    /content/ai_reliability_monitor.py

To launch it from a terminal, use:

    streamlit run /content/ai_reliability_monitor.py

The prototype will open in a browser when Streamlit is
running.

IMPORTANT:
The Streamlit app uses the existing OOF analysis results.
It does not retrain the models or modify the previous analysis.
""")

else:

    print("""
Streamlit is not currently available in this environment.

The application code has still been prepared at:

    /content/ai_reliability_monitor.py

The next simplest browser-compatible option is to install
Streamlit and then launch the application with:

    !pip install streamlit

followed by:

    !streamlit run /content/ai_reliability_monitor.py

The prototype uses the existing OOF analysis results.
It does not retrain the models or modify the previous analysis.
""")


# ============================================================
# 19. SIMPLE USER INSTRUCTIONS
# ============================================================

print("""
HOW TO USE THE PROTOTYPE
------------------------

1. Launch the Streamlit application.
2. Select one of the 569 observations.
3. Review the four model predictions.
4. Review each model's probability of Benign.
5. Review each model's uncertainty.
6. Check whether all four models agree.
7. Review the six pairwise disagreement values.
8. Review average and maximum disagreement.
9. Review the combined MODEL RELIABILITY RISK SCORE.
10. Review the risk quartile and reliability flag.
11. Use the probability visualization to see how closely
    the four model probabilities are aligned.

The HIGH RISK / HUMAN REVIEW flag indicates that the
observation falls within the highest-risk quartile according
to the research framework.

It does NOT mean that the prediction is known to be wrong.
""")
