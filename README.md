📊 AI-Powered Sales Prediction Dashboard
An interactive web dashboard that predicts future sales using LSTM (Long Short-Term Memory) deep learning model and visualizes insights using Streamlit.

This project demonstrates time-series forecasting, deep learning implementation, and real-time dashboard deployment.

🚀 Features
📈 Historical sales visualization

🤖 LSTM-based sales prediction

🔮 Next-day forecast generation

📊 Interactive Streamlit dashboard

🧠 Data preprocessing with MinMaxScaler

🛠️ Tech Stack
Python

TensorFlow / Keras

Pandas & NumPy

Scikit-learn

Matplotlib

Streamlit

📂 Project Structure
AI-Sales-Prediction-Dashboard/
│
├── app.py              # Streamlit dashboard
├── train_model.py      # Model training script
├── sales_data.csv      # Dataset
├── model.h5            # Trained LSTM model
├── scaler.save         # Saved scaler
├── requirements.txt
└── README.md
AI-Sales-Prediction-Dashboard

1...Create Virtual Environment (Recommended)
python -m venv venv

2....Activate it:
venv\Scripts\activate

3... Install Dependencies
pip install -r requirements.txt

4...Train the Model
python train_model.py


->This will generate: model.h5
                      scaler.save

▶️ Run the Dashboard
streamlit run app.py
The app will open automatically in your browser.

📊 Future Improvements
Multi-day forecasting

Upload custom CSV feature

Model comparison (ARIMA vs LSTM)

Deployment on Streamlit Cloud

Based on the sales_data.csv dataset the prediction is done. 

🎯 Resume Description

Developed an AI-powered sales forecasting dashboard using LSTM deep learning model and Streamlit to predict future revenue trends and visualize business insights.
