import streamlit as st
import pickle
import numpy as np
import os


model_path = "model/health_model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found! Please run 'Model_train.py' first to train and save the model.")
else:
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    
    st.title("🩺 Health Detection System")
    st.write("This system predicts your risk of diabetes based on key health parameters.")

    
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 200, 110)
    blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin Level", 0, 900, 80)
    bmi = st.slider("BMI (Body Mass Index)", 0.0, 67.0, 20.0)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 25)

  
    if st.button("Predict"):
       
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, diabetes_pedigree, age]])

        
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠ The person is likely to have diabetes.")
        else:

            st.success("✅ The person is unlikely to have diabetes.")
