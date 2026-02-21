import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

st.title("📊 AI Sales Prediction Dashboard")

# Load model & scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.save")

# Load dataset
df = pd.read_csv("sales_data.csv")
st.subheader("📈 Historical Sales Data")
st.line_chart(df["Sales"])

# Prepare last 3 values for prediction
data = df["Sales"].values.reshape(-1, 1)
scaled_data = scaler.transform(data)

last_3 = scaled_data[-3:]
X_test = np.reshape(last_3, (1, 3, 1))

prediction = model.predict(X_test)
prediction = scaler.inverse_transform(prediction)

st.subheader("🔮 Next Day Predicted Sales")
st.write(f"Predicted Sales: {prediction[0][0]:.2f}")