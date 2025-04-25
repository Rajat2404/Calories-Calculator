import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('calorie_model.pkl')
scaler = joblib.load('calorie_scaler.pkl')

# Mapping options
meal_type_map = {
    "Breakfast": 0,
    "Lunch": 1,
    "Dinner": 2
}

category_map = {
    "Beverages": 0,
    "Dessert": 1,
    "Main Course": 2,
    "Snack": 3,
    "Soup": 4
}

st.title("Daily Calorie Predictor")

# Input fields
protein = st.number_input("Protein (g)", min_value=0.0)
carbs = st.number_input("Carbohydrates (g)", min_value=0.0)
fat = st.number_input("Fat (g)", min_value=0.0)
fiber = st.number_input("Fiber (g)", min_value=0.0)
sugars = st.number_input("Sugars (g)", min_value=0.0)
sodium = st.number_input("Sodium (mg)", min_value=0.0)
cholesterol = st.number_input("Cholesterol (mg)", min_value=0.0)

# Dropdowns
meal_type = st.selectbox("Meal Type", list(meal_type_map.keys()))
category = st.selectbox("Category", list(category_map.keys()))

if st.button("Calculate Calories"):
    try:
        input_data = np.array([[protein, carbs, fat, fiber, sugars, sodium, cholesterol,
                                meal_type_map[meal_type], category_map[category]]])
        input_scaled = scaler.transform(input_data)
        result = model.predict(input_scaled)[0]
        st.success(f"Estimated Calorie Intake: {result:.2f} kcal")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
