## Customer Loan Prediction Project README

In the realm of financial services, the ability to accurately predict loan defaults is of paramount importance. This project addresses this challenge by leveraging machine learning techniques to forecast whether a customer will default on a loan, based on various factors, including credit history and demographic information.

## Dataset

The project utilizes two key datasets:

- **Customer_Loan_Prediction_Train.csv**: This dataset comprises 814 meticulously curated customer records, encompassing crucial attributes such as gender, marital status, income details, loan parameters, credit history, and property area.

- **Customer_Loan_Prediction_Test.csv**: Serving as the evaluation ground, this dataset contains 367 records mirroring the structure of the training set but without the target variable (default status).

## Algorithm and Model Evaluation

The heart of the project lies in the development of a sophisticated machine learning algorithm, meticulously crafted within the `Customer Loan Prediction.ipynb`. Key highlights include:

- Utilization of pandas and scikit-learn libraries for data processing and model development.

- Fine-tuning of hyperparameters to optimize model performance.

- Evaluation of multiple classifiers, including logistic regression, multinomial Naive Bayes, and Bernoulli Naive Bayes.

- Selection of the Bernoulli Naive Bayes model as the top performer, achieving an impressive accuracy of **79.6796%**.

## Predictions

Armed with the optimized algorithm, predictions were made on the test dataset. The outcome of this predictive journey resulted in the creation of a new file named `Customer_Loan_Prediction_Test_Final.csv`, housing the forecasted target variable for each customer record.

## Analysis

A comprehensive analysis of the training dataset was conducted to unravel insights into the factors influencing loan approval likelihood. Key components of this analysis include:

- In-depth exploration of various features using pandas and visualization tools.

- Unearthing valuable insights into the relationships between different attributes and loan default likelihood.

## Additional Files

- **Customer_Loan_Prediction_Test.csv**: The test dataset without output, utilized for predictive modeling.

- **Customer_Loan_Prediction_Train.csv**: The cornerstone of the project, providing data for algorithm development and analysis.

- **Customer_Loan_Prediction_Test_Final.csv**: The output file containing predicted defaults generated by the algorithm.

This project represents a comprehensive approach to predictive modeling in the domain of loan default prediction, combining technical expertise with a deep understanding of the underlying data to deliver actionable insights.
