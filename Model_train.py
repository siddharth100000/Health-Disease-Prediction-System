import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


if not os.path.exists("model"):
    os.makedirs("model")


url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)


X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier()
model.fit(X_train, y_train)


with open("model/health_model.pkl", "wb") as f:
    pickle.dump(model, f)

st.success("✅ Model trained and saved to model/health_model.pkl")

