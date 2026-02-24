# TASK 4 - IRIS FLOWER CLASSIFICATION
# Machine Learning Model using Streamlit


# Import required libraries
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Title of Web App

st.title("Iris Flower Classification App")

st.write("This app predicts Iris flower species using Machine Learning.")

# Load Dataset

# Using built-in Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Machine Learning Model

model = LogisticRegression()

model.fit(X_train, y_train)


# User Input Section

st.subheader("Enter Flower Measurements")

sepal_length = st.slider("Sepal Length", 4.0,8.0,5.5)
sepal_width = st.slider("Sepal Width", 2.0,4.5,3.0)
petal_length = st.slider("Petal Length", 1.0,7.0,4.0)
petal_width = st.slider("Petal Width", 0.1,2.5,1.0)

# Prediction Button

if st.button("Predict Species"):

    # Scale user input
    input_data = scaler.transform([
        [sepal_length,
         sepal_width,
         petal_length,
         petal_width]
    ])

    prediction = model.predict(input_data)

    species_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        "Predicted Species: "
        + species_names[prediction[0]]
    )

    # Data Visualization
   
    st.subheader("Data Visualization")

    fig, ax = plt.subplots()

    colors = ['red','green','blue']
    labels = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    for i in range(3):

        ax.scatter(
            X[y==i,0],   # Sepal Length
            X[y==i,2],   # Petal Length
            c=colors[i],
            label=labels[i]
        )

    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Petal Length")

    ax.legend()

    st.pyplot(fig)