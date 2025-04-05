import streamlit as st
import pandas as pd
import joblib

# Load encoders, scaler, and model
label_encoders = joblib.load("encoder.joblib")
scaler = joblib.load("scaler.joblib")
model = joblib.load("best_regression_model.joblib")

# Define valid items
valid_items = [
    'Maize', 'Potatoes', 'Rice', 'Sorghum', 'Soybeans', 'Wheat',
    'Cassava', 'Sweet potatoes', 'Yams'
]

valid_areas = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia',
    'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus',
    'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi',
    'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'Colombia',
    'Croatia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt',
    'El Salvador', 'Eritrea', 'Estonia', 'Finland', 'France', 'Germany',
    'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
    'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy', 'Jamaica',
    'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon', 'Lesotho', 'Libya',
    'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania',
    'Mauritius', 'Mexico', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia',
    'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
    'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal', 'Qatar',
    'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia',
    'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden',
    'Switzerland', 'Tajikistan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda',
    'Ukraine', 'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe'
]

numerical_features = ['Year', 'average_rain_fall_mm_per_year', 'Pesticides_tonnes', 'avg_temp']
categorical_features = ['Area', 'Item']

# Streamlit UI
st.title("🌾 Crop Yield Prediction App")
st.markdown("Fill in the details below to predict yield (hg/ha):")

area = st.selectbox("🌍 Select Area", valid_areas)
item = st.selectbox("🌱 Select Crop Item", valid_items)
year = st.number_input("📅 Year", min_value=2000, max_value=2100, value=2024)
rainfall = st.number_input("🌧️ Average Rainfall (mm/year)", value=1000.0)
pesticides = st.number_input("🧪 Pesticides Used (tonnes)", value=1000.0)
avg_temp = st.number_input("🌡️ Average Temperature (°C)", value=24.0)

if st.button("Predict Yield"):
    try:
        # Create DataFrame
        input_df = pd.DataFrame({
            'Area': [area],
            'Item': [item],
            'Year': [year],
            'average_rain_fall_mm_per_year': [rainfall],
            'Pesticides_tonnes': [pesticides],
            'avg_temp': [avg_temp]
        })

        # Encode categorical features
        for col in categorical_features:
            if col in label_encoders:
                encoder = label_encoders[col]
                if input_df[col][0] not in encoder.classes_:
                    st.error(f"❌ '{input_df[col][0]}' is not a known category for '{col}'")
                    st.stop()
                input_df[col] = encoder.transform(input_df[col])
            else:
                st.error(f"❌ No encoder found for column '{col}'")
                st.stop()

        # Scale numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted Yield: **{prediction:.2f} hg/ha**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
