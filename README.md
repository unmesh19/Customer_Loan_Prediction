# **Customer Loan Prediction Project README**

## **Project Overview**
The goal of this project is to predict whether a customer will be approved for a loan based on various features, including demographic and financial data. The project involves **data analysis** and the use of machine learning algorithms to classify loan approval (`Loan_Status`).

## **Dataset**
The project uses two key datasets:

- **Customer_Loan_Prediction_Train.csv**: Contains 814 records with demographic and financial information along with the target variable (`Loan_Status`).
  
- **Customer_Loan_Prediction_Test.csv**: Contains 367 records without the target variable. This dataset is used for model evaluation and predictions.

## **Project Flow**

### **1. Data Preprocessing**
- **Missing Data Handling**: Missing values in categorical features (e.g., gender, marital status) were filled with the **mode** (most frequent value) of each column.
- **Feature Engineering**: Categorical variables such as `Gender`, `Married`, `Education`, etc., were encoded using **one-hot encoding** to convert them into numerical values. Continuous variables like `ApplicantIncome` and `LoanAmount` were normalized using **Min-Max scaling** to ensure all features are on a similar scale.

### **2. Exploratory Data Analysis (EDA)**
Exploratory Data Analysis (EDA) was performed to identify patterns and relationships between features and loan approval (`Loan_Status`). Key insights include:
- **Gender**: There is no significant impact of gender on loan approval, as both males and females have similar approval rates.
- **Marital Status**: Married individuals are more likely to be approved for loans, suggesting that marital status may influence loan eligibility.
- **Dependents**: Customers with dependents tend to apply for loans more frequently, indicating that larger families may have higher financial needs.
- **Education**: Graduates are more likely to obtain loans, which could be due to better job prospects and income stability associated with higher education.
- **Property Area**: Individuals from **semi-urban areas** are more likely to be approved for loans compared to those from urban or rural areas. This suggests that economic conditions in semi-urban areas might be more conducive to loan approval.

### **3. Model Training and Evaluation**
Several machine learning models were trained to predict loan approval:
- **Logistic Regression**: Used as the initial model for classification.
- **Naive Bayes Models**: Both **Multinomial Naive Bayes** and **Bernoulli Naive Bayes** were evaluated.
  - **Bernoulli Naive Bayes** performed the best, achieving an accuracy of **79.68%** on the test data.
- **Hyperparameter Tuning**: **GridSearchCV** was used to optimize the hyperparameters of the **Bernoulli Naive Bayes** model, improving its performance.

### **4. Cross-Validation**
To ensure the model's generalizability and to prevent overfitting, **cross-validation** was applied. We used **5-fold cross-validation**, which provided a more reliable estimate of the model's performance by training and evaluating it across multiple subsets of the data.

### **5. Final Predictions**
The trained model was applied to the **test dataset** (`Customer_Loan_Prediction_Test.csv`) to predict loan approval (`Loan_Status`). The predicted results were stored in a new file (`Customer_Loan_Prediction_Test_Final.csv`), containing the loan status for each customer in the test set.

## **Key Findings and Insights**
- **Marital status**, **education**, and **property area** were identified as important factors influencing loan approval.
- **Gender** and **self-employment status** did not have a significant effect on loan approval, indicating that financial institutions may prioritize other factors like income, dependents, and property area.

## **Technologies Used**
- **Python** (pandas, scikit-learn, Matplotlib)
- **Machine Learning**:
  - Logistic Regression
  - Bernoulli Naive Bayes
- **Data Preprocessing**:
  - One-Hot Encoding
  - Min-Max Scaling
- **Model Evaluation**:
  - GridSearchCV for hyperparameter tuning
  - Cross-validation for performance assessment

## **Conclusion**
This project demonstrates the use of **machine learning** techniques for predicting loan approval based on demographic and financial data. The model achieved an accuracy of **79.68%**, providing valuable insights into the factors influencing loan approval. The use of **cross-validation** ensured that the model is robust and generalizable, making it suitable for real-world applications in financial institutions.