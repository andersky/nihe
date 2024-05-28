import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('Titantic.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction function
def predict_survival(PassengerId, sex, age, sibsp, parch, fare, embarked, pclass, family):
    # Create a DataFrame for a single sample
    df = pd.DataFrame({
        'PassengerId': [PassengerId],
        'Sex': [1 if sex == 'male' else 0],
        'Age': [float(age)],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [float(fare)],
        'Embarked_C': [1 if embarked == 'C' else 0],
        'Embarked_Q': [1 if embarked == 'Q' else 0],
        'Embarked_S': [1 if embarked == 'S' else 0],
        'Pclass_1': [1 if pclass == '1' else 0],
        'Pclass_2': [1 if pclass == '2' else 0],
        'Pclass_3': [1 if pclass == '3' else 0],
        'family': [family]
    })
    # Use the model to get the survival probability
    probabilities = model.predict_proba(df)
    survival_probability = probabilities[0][1]
    return f"Survival Probability: {survival_probability:.2%}"

# Create the Streamlit interface
st.title("Titanic Survival Prediction")
st.write("Input the details to predict the survival probability on the Titanic.")

PassengerId = st.number_input("PassengerId", value=1)
sex = st.selectbox("Sex", options=['male', 'female'])
age = st.number_input("Age", value=30.0)
sibsp = st.number_input("Number of Siblings/Spouses", value=0)
parch = st.number_input("Number of Parents/Children", value=0)
fare = st.number_input("Fare", value=32.2)
embarked = st.radio("Embarked", options=['C', 'Q', 'S'])
pclass = st.radio("Pclass", options=['1', '2', '3'])
family = st.number_input("family", value=1)

if st.button("Predict"):
    prediction = predict_survival(PassengerId, sex, age, sibsp, parch, fare, embarked, pclass, family)
    st.write(prediction)
