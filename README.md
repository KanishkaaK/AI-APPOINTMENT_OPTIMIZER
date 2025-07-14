# AI Appointment Availability Predictor

This project uses machine learning to predict the likelihood that a patient will miss or attend their appointment. The goal is to help clinics optimize their scheduling and reduce no-shows.

## Features

* Allows users to select doctor, appointment time, and day of the week
* Collects additional patient details: age, gender, past missed appointments, and distance from clinic
* Option to record contact number verification
* Predicts whether an appointment is likely to be attended or missed
* Generates a report of predictions in CSV format

## Dataset

The model was trained on a synthetic dataset of appointments with the following fields:

* Doctor ID (D001 to D010)
* Appointment Type (Checkup, Consultation, Test, Follow-up)
* Appointment Time (6:00 AM to 8:00 PM, in half-hour intervals)
* Day of the Week
* Delay (in minutes)
* Patient Age
* Gender (Male, Female, Other)
* Past Missed Appointments Count
* Distance from Clinic (in km)
* Contact Number Provided (verified or not)
* Target: Missed or Attended

## How it works

1. The model is trained using a RandomForestClassifier on the features above.
2. The app is built with Streamlit and allows users to enter new appointment details.
3. The trained model predicts the availability of the appointment.
4. The results are logged and can be downloaded for analysis.

## How to Run

1. Clone the repository
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Run the app:

   ```
   streamlit run "AI app.py"
   ```

## Model Performance

The model was trained on synthetic data and achieved an accuracy of **47.52%** on the test set.
For improved performance in real scenarios, a larger and more realistic dataset is recommended.





## 📄 License
This project is licensed under a custom copyright license.  
Usage, reproduction, or distribution without permission is strictly prohibited.  
See `LICENSE` file for full terms.
