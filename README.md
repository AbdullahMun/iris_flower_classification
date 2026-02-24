**Iris Flower Classification Project**
**Machine Learning Internship – Month 2**
**Task 4: Iris Flower Classification**

**Project Overview**

This project implements a machine learning model to classify iris flowers into different species based on their physical measurements. The goal of this task is to build a classification model that can correctly identify the species of an iris flower using features such as sepal length, sepal width, petal length, and petal width.

The project was developed as part of the Machine Learning Internship (Month 2).


**Dataset Description**

The Iris dataset is a well-known dataset in machine learning and is included in the Scikit-learn library.

The dataset contains 150 samples of iris flowers divided into three species:

1) Setosa
2) Versicolor
3) Virginica

Each sample contains the following features:

• Sepal Length
• Sepal Width
• Petal Length
• Petal Width

Target Variable:

• Species of Iris Flower


**Data Preprocessing**

The dataset was loaded using the Scikit-learn library. The data was already clean and did not contain missing values, so minimal preprocessing was required.

The dataset was divided into training and testing sets using an 80-20 split to evaluate model performance.


**Machine Learning Model**

A Logistic Regression model was used for classification.

Logistic Regression was chosen because it is simple, efficient, and works well for classification problems with well-separated classes such as the Iris dataset.

The model was trained using the training dataset and then tested on unseen data.



**Model Evaluation**

The model performance was evaluated using:

• Accuracy Score
• Cross Validation Accuracy
• Classification Report

Results:

Test Accuracy ≈ 98%

Cross Validation Accuracy ≈ 97%

The classification report shows high precision and recall values, indicating that the model can correctly classify iris flowers with very few errors.



**Data Visualization**

Several visualizations were created to better understand the dataset:

• Scatter Plot of Petal Length vs Petal Width
• Correlation Heatmap

These visualizations helped in understanding the relationships between different features.



**Streamlit Application**

A Streamlit web application was developed to make the model interactive.

The user can enter flower measurements and the application predicts the species of the iris flower.

To run the application:

1) Activate Virtual Environment

venv\Scripts\activate

2) Install Required Libraries

pip install -r requirements.txt

3) Run Streamlit Application

streamlit run app.py



**Technologies Used**

• Python
• Scikit-learn
• Pandas
• NumPy
• Matplotlib
• Seaborn
• Streamlit



**Project Structure**

Task4_Iris/

app.py
iris.ipynb
iris_model.pkl
requirements.txt
README.md



**Conclusion**

This project demonstrates how machine learning can be used to classify iris flowers based on their physical characteristics. The Logistic Regression model achieved high accuracy and reliable performance.

The Streamlit application allows users to interact with the model and make real-time predictions.