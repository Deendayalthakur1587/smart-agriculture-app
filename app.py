import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
le_crop = pickle.load(open("le_crop.pkl", "rb"))
le_soil = pickle.load(open("le_soil.pkl", "rb"))
le_stage = pickle.load(open("le_stage.pkl", "rb"))

# Page config
st.set_page_config(page_title="Smart Agriculture AI", layout="centered")

# Title
st.title("🌱 Smart Agriculture IoT AI System")
st.markdown("### 🚀 Predict crop condition using IoT sensor data")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.info("This AI model predicts agricultural conditions using environmental data like moisture, temperature, and humidity.")

# Input UI
st.subheader("📥 Enter Input Values")

col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("🌾 Select Crop", le_crop.classes_)
    soil = st.selectbox("🧱 Soil Type", le_soil.classes_)
    stage = st.selectbox("🌿 Seedling Stage", le_stage.classes_)

with col2:
    moi = st.slider("💧 Moisture Index (MOI)", 0.0, 100.0)
    temp = st.slider("🌡 Temperature (°C)", 0.0, 50.0)
    humidity = st.slider("🌫 Humidity (%)", 0.0, 100.0)

# Predict button
if st.button("🔍 Predict"):

    # Encode inputs
    input_data = np.array([[
        le_crop.transform([crop])[0],
        le_soil.transform([soil])[0],
        le_stage.transform([stage])[0],
        moi,
        temp,
        humidity
    ]])

    prediction = model.predict(input_data)

    st.success(f"✅ Prediction Result: {prediction[0]}")

    # Explanation
    st.subheader("📊 Explanation")
    st.write("This prediction is based on:")
    st.write(f"- Crop: {crop}")
    st.write(f"- Soil Type: {soil}")
    st.write(f"- Growth Stage: {stage}")
    st.write(f"- Moisture: {moi}")
    st.write(f"- Temperature: {temp}")
    st.write(f"- Humidity: {humidity}")

    st.info("💡 Recommendation: Maintain optimal moisture and temperature for better yield.")