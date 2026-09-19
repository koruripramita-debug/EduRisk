import streamlit as st
import json
import numpy as np
import requests

# Load portable Random Forest
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

    return np.mean(predictions)


# Page configuration
st.set_page_config(
    page_title="EduRisk - Student Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

# Sidebar
with st.sidebar:

    st.header("🎓 EduRisk")

    st.write("### About")

    st.write(
        "EduRisk is a machine learning application "
        "that predicts student performance and identifies "
        "academic risk."
    )

    st.write("### Features")

    st.write("✅ Performance Prediction")
    st.write("✅ Academic Risk Detection")
    st.write("✅ Personalized Recommendation")

    st.divider()

    st.caption("Built with Python, Machine Learning & Streamlit")

# Header
st.title("🎓 EduRisk")

st.markdown(
    "### Student Performance & Academic Risk Predictor"
)

st.write(
    "Predict student performance and identify academic risk "
    "using machine learning."
)

st.divider()

st.subheader("🔄 How EduRisk Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("### 1️⃣")
    st.write("Enter Student Data")
    st.caption("Study, attendance, marks, assignments and sleep.")

with step2:
    st.markdown("### 2️⃣")
    st.write("ML Prediction")
    st.caption("The trained Random Forest model analyzes the data.")

with step3:
    st.markdown("### 3️⃣")
    st.write("Risk Detection")
    st.caption("EduRisk identifies the academic risk level.")

with step4:
    st.markdown("### 4️⃣")
    st.write("Recommendation")
    st.caption("The student receives a simple improvement suggestion.")

# Student Information
st.subheader("👨‍🎓 Student Information")

col1, col2 = st.columns(2)


with col1:

    study_hours = st.number_input(
        "📚 Study Hours",
        min_value=1,
        max_value=24,
        value=6
    )

    attendance = st.number_input(
        "📅 Attendance (%)",
        min_value=0,
        max_value=100,
        value=85
    )

    previous_marks = st.number_input(
        "📝 Previous Marks",
        min_value=0,
        max_value=100,
        value=75
    )


with col2:

    assignments = st.number_input(
        "📋 Assignment Score",
        min_value=0,
        max_value=100,
        value=80
    )

    sleep_hours = st.number_input(
        "😴 Sleep Hours",
        min_value=1,
        max_value=24,
        value=7
    )


# Prediction
if st.button(
    "🚀 Predict Student Performance",
    use_container_width=True
):

    payload = {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignments": assignments,
        "sleep_hours": sleep_hours
    }

    response = requests.post(
        "http://127.0.0.1:3000/predict",
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    result_body = result.get("body", result)

    if isinstance(result_body, str):
        result_body = json.loads(result_body)

    prediction = float(result_body["predicted_marks"])
    risk = result_body["risk"]
    recommendation = result_body["recommendation"]

    # Results
    st.divider()

    st.subheader("📊 Prediction Results")

    result1, result2 = st.columns(2)


    with result1:

        st.metric(
            "🎯 Predicted Final Marks",
            f"{prediction:.2f}"
        )


    with result2:

        st.metric(
            "⚠️ Academic Risk Level",
            risk
        )


    st.write("")

    st.subheader("💡 Recommendation")


    if risk == "Low Risk":

        st.success(recommendation)

    elif risk == "Medium Risk":

        st.warning(recommendation)

    else:

        st.error(recommendation)
# Model Information
st.divider()

st.subheader("🤖 About the ML Model")

model1, model2, model3 = st.columns(3)

with model1:
    st.metric("🌳 Model", "Random Forest")

with model2:
    st.metric("📏 MAE", "3.8")

with model3:
    st.metric("📊 R² Score", "0.76")

st.caption(
    "The model was trained using student academic and study-related data."
)

st.caption(
    "The model was trained using student academic and study-related data."
)


# Risk Level Guide
st.divider()

st.subheader("⚠️ Academic Risk Guide")

risk1, risk2, risk3 = st.columns(3)

with risk1:
    st.success("### 🟢 Low Risk")
    st.write("Predicted score: 80 or above")
    st.caption("Maintain current study habits.")

with risk2:
    st.warning("### 🟡 Medium Risk")
    st.write("Predicted score: 60–79.99")
    st.caption("Improve study time, attendance and assignments.")

with risk3:
    st.error("### 🔴 High Risk")
    st.write("Predicted score: Below 60")
    st.caption("Consider additional academic support.")


# Footer
st.divider()

st.markdown("### 👥 ")
# Footer
st.divider()

st.markdown("### 👥 ")

team1, team2 = st.columns(2)

with team1:
    st.write("👩‍💻 ****")
    st.caption("Machine Learning, Data & Website Development")

with team2:
    st.write("👩‍💻 **Roshni Rajak**")
    st.caption("")