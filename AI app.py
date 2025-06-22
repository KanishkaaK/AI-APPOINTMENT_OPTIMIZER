# AI app.py — CORRECTED

import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model and encoders
model = joblib.load('miss_model.joblib')
le_appointment_type = joblib.load('label_encoder_appointment_type.joblib')
le_doctor = joblib.load('label_encoder_doctor.joblib')
le_gender = joblib.load('label_encoder_gender.joblib')

# Title
st.title("AI Appointment Availability Predictor")

# Convert appointment time string to number
def convert_hour_str(hour_str):
    h, m = hour_str.split(':')
    h = int(h)
    if 'PM' in hour_str and h != 12:
        h += 12
    if 'AM' in hour_str and h == 12:
        h = 0
    return h + (0.5 if '30' in hour_str else 0)

# Inputs
selected_doctor = st.selectbox("Select Doctor", le_doctor.classes_)
doctor_encoded = le_doctor.transform([selected_doctor])[0]

appointment_time = st.selectbox(
    "Select Appointment Time (6 AM to 8 PM)",
    [
        "06:00 AM", "06:30 AM", "07:00 AM", "07:30 AM", "08:00 AM", "08:30 AM", "09:00 AM", "09:30 AM",
        "10:00 AM", "10:30 AM", "11:00 AM", "11:30 AM", "12:00 PM", "12:30 PM", "01:00 PM", "01:30 PM",
        "02:00 PM", "02:30 PM", "03:00 PM", "03:30 PM", "04:00 PM", "04:30 PM", "05:00 PM", "05:30 PM",
        "06:00 PM", "06:30 PM", "07:00 PM", "07:30 PM", "08:00 PM"
    ]
)

appointment_hour_num = convert_hour_str(appointment_time)

appointment_day_of_week = st.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x]
)

delay_mins = st.number_input("Expected Delay (mins)", min_value=0, value=0)

appointment_type = st.selectbox("Appointment Type", le_appointment_type.classes_)
appointment_type_encoded = le_appointment_type.transform([appointment_type])[0]

patient_age = st.slider("Patient Age", 1, 100, 30)

patient_gender = st.selectbox("Patient Gender", le_gender.classes_)
gender_encoded = le_gender.transform([patient_gender])[0]

past_miss_count = st.number_input("Past Miss Count", min_value=0, value=0)

distance_from_clinic_km = st.number_input("Distance from Clinic (km)", min_value=0, value=2)

contact_number = st.text_input("Contact Number (optional)")
contact_number_verified = 1 if contact_number.strip() != '' else 0

# PREDICT
if st.button("Predict Availability"):

    input_data = pd.DataFrame(
        [[doctor_encoded, appointment_hour_num, appointment_day_of_week, delay_mins, appointment_type_encoded,
          patient_age, gender_encoded, past_miss_count, distance_from_clinic_km, contact_number_verified]],
        columns=[
            'doctor_id_encoded',
            'appointment_hour_num',
            'appointment_day_of_week',
            'delay_mins',
            'appointment_type_encoded',
            'patient_age',
            'gender_encoded',
            'past_miss_count',
            'distance_from_clinic_km',
            'contact_number_verified'
        ]
    )

    # Predict
    predicted_prob = model.predict_proba(input_data)[0][1]

    # Show result
    if predicted_prob < 0.3:
        st.success("Appointment Available — likely to be attended.")
    elif 0.3 <= predicted_prob < 0.6:
        st.warning("Medium Risk — consider confirming with patient.")
    else:
        st.error("Not Available — high risk of no-show.")

    
    # Log prediction
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'doctor_id': selected_doctor,
        'appointment_hour': appointment_hour_num,
        'day_of_week': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][appointment_day_of_week],
        'delay_mins': delay_mins,
        'appointment_type': appointment_type,
        'patient_age': patient_age,
        'patient_gender': patient_gender,
        'past_miss_count': past_miss_count,
        'distance_from_clinic_km': distance_from_clinic_km,
        'contact_number': contact_number,
        'availability': "Available" if predicted_prob < 0.3 else "Risk",
        'suggestion': suggestion
    }
    
    log_df = pd.DataFrame([log_entry])
    
    if os.path.exists("Confirmation_report.csv"):
        log_df.to_csv("Confirmation_report.csv", mode='a', header=False, index=False)
    else:
        log_df.to_csv("Confirmation_report.csv", index=False)
    
    st.info(" Confirmed_Reort")

# --- Download log ---
if os.path.exists("Confirmation_report.csv"):
    with open("Confirmation_report.csv", "rb") as f:
        st.download_button("📥 Download Confirmation_report (CSV)", f, file_name="Confirmation_report.csv")
