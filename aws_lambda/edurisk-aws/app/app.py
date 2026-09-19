import json
import numpy as np


# Load the trained Random Forest model data
with open("model_data.json", "r") as f:
    forest = json.load(f)


# Predict using one decision tree
def predict_tree(tree, features):

    node = 0

    while tree["children_left"][node] != -1:

        feature_index = tree["feature"][node]
        threshold = tree["threshold"][node]

        if features[feature_index] <= threshold:
            node = tree["children_left"][node]
        else:
            node = tree["children_right"][node]

    return tree["value"][node]


# Predict using all trees
def predict_student(features):

    predictions = []

    for tree in forest:
        predictions.append(
            predict_tree(tree, features)
        )

    return float(np.mean(predictions))


def get_risk_level(score):

    if score >= 80:
        return "Low Risk"

    elif score >= 60:
        return "Medium Risk"

    else:
        return "High Risk"


def get_recommendation(risk):

    if risk == "Low Risk":

        return "Keep up the good work and maintain your current study habits."

    elif risk == "Medium Risk":

        return "Increase study time, improve attendance, and focus on assignments."

    else:

        return "Increase study time, improve attendance, and seek academic support."


def lambda_handler(event, context):

    # Get student data
    body = event.get("body", event)

    if isinstance(body, str):
        body = json.loads(body)

    study_hours = float(body["study_hours"])
    attendance = float(body["attendance"])
    previous_marks = float(body["previous_marks"])
    assignments = float(body["assignments"])
    sleep_hours = float(body["sleep_hours"])

    # Feature order must match the trained model
    features = [
        study_hours,
        attendance,
        previous_marks,
        assignments,
        sleep_hours
    ]

    # Real Random Forest prediction
    prediction = predict_student(features)

    # Risk
    risk = get_risk_level(prediction)

    # Recommendation
    recommendation = get_recommendation(risk)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "predicted_marks": round(prediction, 2),
            "risk": risk,
            "recommendation": recommendation
        })
    }