Heart Disease Prediction Project
Project Overview

The project aims to predict the likelihood of a patient developing heart disease within ten years based on clinical and lifestyle features. Using machine learning models such as logistic regression, decision trees and random forests, the project analyzes patient data to provide accurate predictions.

dataset

The dataset includes clinical, demographic and lifestyle data for patients.

Key features include:

Age – Age of the patient

Male – Gender of the patient (1 = male, 0 = female)

cigsPerDay – average number of cigarettes smoked per day

BPMeds - Is the patient taking blood pressure medication

Prevalence of Stroke - History of Stroke

PrevalenceHyper - History of high blood pressure

diabetes - diabetic condition

totchol - total cholesterol

sysBP - systolic blood pressure

BMI - Body Mass Index

heart rate - heart rate

glucose - blood glucose level

CHD – target variable: 1 if the patient develops coronary heart disease within ten years, 0 otherwise.

Source: Framingham Heart Study Dataset

project phase
1. Data Preprocessing

Removed unnecessary columns like education, currentsmoker, diabetes etc.

TenYearCHD was renamed CHD.

Outliers were removed in sysBP, BMI, heart rate, glucose, totchol.

Missing values ​​were imputed using the most frequent approach.

Standardized numerical features (age, totchole, sysBP, BMI, heart rate, glucose, cigarettes per day) using StandardScaler.

2. Data Visualization

Count plot of gender distribution (male).

Calculate CHD incidence by gender.

Heatmap of correlation between features.

3. Model Training

Logistic Regression – A linear model for binary classification.

Decision Tree Classifier – A tree-based model with max_depth=3.

Random Forest Classifier – An ensemble of decision trees (n_estimator=3).

4. Model evaluation
Accuracy scores were used to evaluate predictions on test data.

The performance of logistic regression, decision tree, and random forest was compared.

how to run a project

Clone the repository or download the dataset.

Make sure you have Python 3.x installed.

Install required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn


Run the Python script:

python heart_disease_prediction.py


Output:

Visualization (plot for gender distribution, CHD and correlation heatmap).

Accuracy scores for each model printed in the console.

used library

numpy - numerical calculation

pandas – data manipulation and analysis

Matplotlib and Seaborn - Data Visualization

Scikit-learn – Machine learning algorithms, preprocessing and evaluation

notes

Make sure all columns are pre-processed correctly before training the model.

Scaling is important for logistic regression to converge properly.

Random forest hyperparameters (n_estimator, max_depth) can be tuned for better performance.

Author

MOHAMMED - Project developed as part of Machine Learning/AI course.
