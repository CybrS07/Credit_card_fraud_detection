"""
Pipeline: load data -> remove excess/duplicate entries -> preprocess ->
handle class imbalance (SMOTE) -> train a classifier -> evaluate.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "creditcard.csv"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape}")

    # Remove excess/duplicate transaction rows so the same transaction
    # isn't counted (and learned from) more than once
    df = df.drop_duplicates()
    print(f"Shape after removing duplicate entries: {df.shape}")
    return df


def preprocess(df):
    # Amount and Time sit on a much larger scale than the anonymised
    # V1-V28 features, so scale them to mean 0 / std 1 to match
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(["Amount", "Time"], axis=1)

    X = df.drop("Class", axis=1)   # features
    y = df["Class"]                # label: 0 = normal, 1 = fraud
    return X, y


def split_and_balance(X, y, test_size=0.2, random_state=42):
    # Stratify so both the train and test sets keep the same fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")

    # Fraud is under 1% of transactions, so a model trained as-is would
    # just learn to predict "normal" every time. SMOTE oversamples the
    # minority (fraud) class in the TRAINING set only, so the test set
    # is left untouched and still reflects real-world imbalance.
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {y_train_res.value_counts().to_dict()}")

    return X_train_res, X_test, y_train_res, y_test


def train_model(X_train, y_train, random_state=42):
    # Why Random Forest, out of all the classifiers we could pick:
    # - Fraud vs. normal transactions is a non-linear, messy boundary in
    #   the 30-feature space -- a linear model (e.g. Logistic Regression)
    #   struggles to separate that as well.
    # - It's an ensemble of many decision trees, so it's far less prone to
    #   overfitting than a single decision tree, while still capturing
    #   complex feature interactions.
    # - It doesn't require feature scaling to work well, and copes fine
    #   with the mix of PCA components + scaled Amount/Time we feed it.
    # - It naturally handles the (now SMOTE-balanced) training data well
    #   and gives useful feature importances for free.
    # - It's fast to train and rarely needs heavy tuning to get solid
    #   results, which suits a first working fraud-detection model.
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of fraud, needed for ROC-AUC

    # Accuracy alone is misleading on this data (predicting "normal" for
    # everything would already score ~99.8%), so we look at it alongside
    # precision/recall/F1, which actually show how well fraud is caught
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc:.4f}")

    # Confusion matrix: how many frauds were caught vs. missed, and how
    # many false alarms were raised
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC curve: trade-off between catching fraud and raising false alarms
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def main():
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_and_balance(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()