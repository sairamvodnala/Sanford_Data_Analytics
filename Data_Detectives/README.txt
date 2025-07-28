
# Sanford Health Claims Analysis Project Overview

This project aims to predict high-dollar and delayed insurance claims using the Sanford Health dataset, focusing on identifying patterns that forecast the influx of claims to manage and allocate resources efficiently.

## Problem 1: Predicting High-Dollar Claims

### Data Loading and Cleaning
- **Data Splitting**: Divided data based on 'High Cost Claim' null values into train_data and test_data(needs prediction).
- **Duplicate Removal**: Removed all duplicate rows, retaining only unique entries.
- **Handling Missing Values**: Dropped columns with more than 50% missing values and replaced nulls in specific columns based on similarities with other columns.

### Data Cleaning and EDA
Columns with high percentages of missing values were removed. 
Specific categorical columns were filled based on the most common values to handle nulls effectively.

### Feature Engineering
Created features to capture the time dynamics of claims processing:
- Claim Processing Time: Days between Service Date and Received Date.
- Claim Payment Time: Days between Received Date and Paid Date.
- Total Claim Duration: Days between Service Date and Paid Date.

### Data Preprocessing
Converted categorical columns to numerical through mapping and frequency encoding for various features such as 'LOB', 'Network Status', 'Ethnicity', and more.

### Feature Selection and Model Building
Employed methods like Feature Importance from Ensemble Models and Recursive Feature Elimination to select significant predictors and chi square test.

considered best 12 features # Claim Category
# Service Type
# Place of Service
# ICD10 Code 3
# ICD10 Code 2
# Claim Processing Time
# Total Claim Duration
# ICD10 Code 1
# Member Age
# Claim Payment Time
# Network Status
# Service Code

### Training
Divided the data set into (train+validation) and test with (70+10) and 20 ratio. (stratified) 

Models applied include Random Forest, SVM, KNN, Logistic Regression, Gradient Boosting, XGBoost, and LightGBM. 
from the result Random Forest, XGBoost, and LightGBM are best performing.
As data is imbalanced, SMOTE was used for addressing class imbalance, applied only on train data for model training.
so considered XGBoost as best model

### Model Selection
Finalized XGBoost as the optimal model based on precision, accuracy, and F1 score. but the model is overfitting, to reduce over fitting we did hyperparameter tuning OPTUNA to 
know best parameters. then trained the model with those parameters and still overfitting so we used regularization. then overfitting is reduced.
next it is evaluated with validation data

model now has consistent performance across validation and test sets, which means overfitting is significantly reduced and the model generalizes well.
and saved to a pickle file.

###Applied test data on model then claims are predicted and saved into csv file.

## Problem 2: Predicting Delayed Claims

### Distribution Inspection and Threshold Setting
Utilized the 95th and 99th percentiles of Claim Payment Time to label claims as "delayed."

### Forecasting Monthly Delayed Claims
Used the Prophet model to forecast delayed claims based on the service date. The process included creating a monthly aggregation of delayed claims, fitting the model, and generating a six-month forecast.

### Insights from Forecast
- **Increasing Trend**: Noted a steady increase in delayed claims suggesting a need for operational adjustments.
- **Seasonal Patterns**: Identified peaks during mid-year and end-year, suggesting higher strain during these periods.
- **Uncertainty in Predictions**: Confidence intervals widen in forecasts, indicating the need for continuous model updates.

## Instructions for Evaluation
To run the analyses:
1. Install dependencies from 'requirements.txt'.
2. Execute the Jupyter Notebook 'Sanford_DA2.ipynb' to replicate the analysis.
3. Review model outputs in the notebook and the saved predictions in 'final_predicted_claims.csv'.


Datasets:
DSU_Dataset is CSV file converted from excel file.

train_data and test_data are two files Divided based on 'High Cost Claim' null values

xgboost_final.pkl is the trained model.

cleaned_data is a preprocessed file before training.

train,test,validation are three csv files for training, testing and evaluation.(these files are stratified and preprocessed)

validation_actual_values and predicted_claims are validation data used for evaluation.

preprocessed_data_for_prediction is preprocessed test_data for predictions (applying this file on model)

final_predicted_claims is a file which contains predictions from the model.

ipynb files:
Sanford_DA_2 is a final ipynb file.
testdatapreprocess2.ipynb file has test_data(needs prediction) preprocessing.



This README provides an overview of the methodologies and approaches used in the project, alongside instructions for replicating the analysis and utilizing the models developed.
