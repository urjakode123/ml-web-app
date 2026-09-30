from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI()


# Load trained ML artifacts
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


class StudentData(BaseModel):
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Tutoring_Sessions: float
    Physical_Activity: float

    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Motivation_Level: str
    Internet_Access: str
    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    Gender: str


@app.get("/health")
def health():
    return {
        "status": "API is running"
    }


@app.post("/predict")
def predict(data: StudentData):

    # Convert incoming JSON into a DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    categorical_columns = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Motivation_Level",
        "Internet_Access",
        "Family_Income",
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Learning_Disabilities",
        "Parental_Education_Level",
        "Distance_from_Home",
        "Gender"
    ]

    # Apply the same categorical encoding used during training
    input_data = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=False
    )

    # Keep exactly the feature columns expected by the trained model
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Apply the trained scaler
    input_scaled = scaler.transform(input_data)

    # Generate prediction
    prediction = model.predict(input_scaled)

    return {
        "predicted_exam_score": float(prediction[0])
    }