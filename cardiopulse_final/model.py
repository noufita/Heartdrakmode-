"""
Data loading, feature engineering, and model training for CardioPulse.
Mirrors heart_risk_notebook.ipynb. Cached so it runs once per session.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

import os
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_disease_risk_dataset_earlymed.csv")

# (column key, human label) — order/spelling must match the CSV columns
SYMPTOMS = [
    ("Chest_Pain", "Chest Pain"), ("Shortness_of_Breath", "Shortness of Breath"),
    ("Fatigue", "Fatigue"), ("Palpitations", "Palpitations"), ("Dizziness", "Dizziness"),
    ("Swelling", "Swelling"), ("Pain_Arms_Jaw_Back", "Pain in Arms/Jaw/Back"),
    ("Cold_Sweats_Nausea", "Cold Sweats / Nausea"),
]
RISKS = [
    ("High_BP", "High Blood Pressure"), ("High_Cholesterol", "High Cholesterol"),
    ("Diabetes", "Diabetes"), ("Smoking", "Smoking"), ("Obesity", "Obesity"),
    ("Sedentary_Lifestyle", "Sedentary Lifestyle"), ("Family_History", "Family History"),
    ("Chronic_Stress", "Chronic Stress"),
]

# the 21-feature order the models are trained on
FEATURE_ORDER = (
    [k for k, _ in SYMPTOMS] + [k for k, _ in RISKS] +
    ["Gender", "Age", "Symptom_Count", "Risk_Factor_Count", "Age_Group"]
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).astype(int)

    sym_cols = [k for k, _ in SYMPTOMS]
    risk_cols = [k for k, _ in RISKS]
    df["Symptom_Count"] = df[sym_cols].sum(axis=1)
    df["Risk_Factor_Count"] = df[risk_cols].sum(axis=1)
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 40, 60, 100],
                             labels=["Young", "Middle", "Senior"]).map(
                                 {"Young": 0, "Middle": 1, "Senior": 2}).astype(int)
    return df


@st.cache_resource(show_spinner="Training models…")
def train_models(df: pd.DataFrame) -> dict:
    y = df["Heart_Risk"]
    X = df[FEATURE_ORDER]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    models, preds = {}, {}

    log = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    models["LogReg"] = log
    preds["LogReg"] = log.predict(X_test_s)

    tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    models["Tree"] = tree
    preds["Tree"] = tree.predict(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    models["RF"] = rf
    preds["RF"] = rf.predict(X_test)

    metrics, confusion = {}, {}
    for k, p in preds.items():
        metrics[k] = {
            "acc": accuracy_score(y_test, p) * 100,
            "precision": precision_score(y_test, p),
            "recall": recall_score(y_test, p),
            "f1": f1_score(y_test, p),
        }
        confusion[k] = confusion_matrix(y_test, p)

    return {"models": models, "scaler": scaler, "metrics": metrics,
            "confusion": confusion, "X_test": X_test, "y_test": y_test}
