import gradio
import joblib
import numpy as np
import joblib

loaded_model = joblib.load("xgboost-model.pkl")

def predict_death_event(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time):
    input_data = np.array([[age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time]])
    prediction = loaded_model.predict(input_data)[0]
    if prediction == 0:
        return "The patient is predicted to survive."
    else:
        return "The patient is predicted to not survive."

inputs=[
        gradio.Number(label="Age"),
        gradio.Radio(["0", "1"], label="Anaemia (0: No, 1: Yes)"),
        gradio.Radio(["0", "1"], label="High Blood Pressure (0: No, 1: Yes)"),
        gradio.Number(label="Creatinine Phosphokinase"),
        gradio.Radio(["0", "1"], label="Diabetes (0: No, 1: Yes)"),
        gradio.Number(label="Ejection Fraction"),
        gradio.Number(label="Platelets"),
        gradio.Radio(["0", "1"], label="Sex (0: Female, 1: Male)"),
        gradio.Number(label="Serum Creatinine"),
        gradio.Number(label="Serum Sodium"),
        gradio.Radio(["0", "1"], label="Smoking (0: No, 1: Yes)"),
        gradio.Number(label="Time (Follow-up Period in days)"),
    ]
# Output response
# YOUR CODE HERE
outputs=gradio.Textbox(label="Prediction")

title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = inputs,
                         outputs = outputs,
                         title = title,
                         description = description,
                         allow_flagging='never')

iface.launch(share = True)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface
