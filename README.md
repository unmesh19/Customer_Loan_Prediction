# **Customer Loan Prediction Project**

## **Project Overview**

In the financial services industry, predicting loan defaults is a crucial aspect of risk management. This project aims to predict whether a customer will default on a loan based on their demographic and financial features. The model is developed using various machine learning techniques, including **Logistic Regression**, **Naive Bayes**, and **Bernoulli Naive Bayes** with **hyperparameter tuning** and **cross-validation** to optimize the performance.

## **Dataset**

The project utilizes two key datasets:

- **Customer_Loan_Prediction_Train.csv**: Contains 814 customer records with attributes such as gender, marital status, income, loan parameters, credit history, and property area.
- **Customer_Loan_Prediction_Test.csv**: The test dataset with 367 records, structured similarly to the training set but without the target variable (**Loan_Status**).

## **Algorithm and Model Evaluation**

### **Key Features**:
- **Libraries**: Utilized **pandas** and **scikit-learn** for data processing, feature engineering, model training, and evaluation.
- **Model Selection**: Evaluated multiple models including **Logistic Regression**, **Multinomial Naive Bayes**, and **Bernoulli Naive Bayes**.
- **Hyperparameter Tuning**: Used **GridSearchCV** with expanded hyperparameters and **Stratified K-Fold Cross-Validation** to optimize the Bernoulli Naive Bayes model.
- **Model Evaluation**: Evaluated models using multiple metrics such as **accuracy**, **F1-score**, **ROC-AUC**, and **cross-validation scores**.

### **Model Tuning**:
- **GridSearchCV**: Expanded to test **5 values per parameter** for `alpha` (smoothing parameter) and `binarize` (threshold for binarization).
- **Best Parameters**: 
  - `alpha = 10.0`
  - `binarize = 0.0`
- **Best ROC-AUC Score**: **76.31%**

### **Cross-Validation**:
- **Stratified K-Fold Cross-Validation** was used with 5 splits to ensure that the class distribution was maintained in each fold.
- **Cross-Validation Metrics**:
  - **Accuracy**: **80.79%**
  - **F1-Score**: **87.53%**
  - **ROC-AUC**: **75.08%**

## **Model Evaluation with Multiple Metrics**

The models were evaluated using a variety of metrics to ensure robustness, especially in the case of imbalanced classes:

- **Precision**: 0.7596
- **Recall**: 0.9875
- **Confusion Matrix**:
  ```
  [[18 25]
   [ 1 79]]
  ```

These metrics highlight the model's ability to correctly predict loan defaults while minimizing false positives.

### **Metrics Explained**:
- **Precision**: Indicates the proportion of true positives among all positive predictions (important for loan approval predictions).
- **Recall**: Measures the proportion of actual loan defaults correctly predicted by the model.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced metric for imbalanced data.
- **ROC-AUC**: Measures the area under the receiver operating characteristic curve, with higher values indicating better classification performance.

## **Predictions**

Using the optimized **Bernoulli Naive Bayes** model, predictions were made on the test dataset. The final predictions were saved in the `Customer_Loan_Prediction_Test_Final.csv` file, which contains the predicted **Loan_Status** (Y/N) for each record.

## **Key Steps in the Project**:

1. **Data Preprocessing**:
   - Loaded and cleaned the dataset, handling missing values and encoding categorical variables.
   - Applied **Min-Max scaling** to numerical features for better model convergence.

2. **Model Training**:
   - Trained **Logistic Regression**, **Multinomial Naive Bayes**, and **Bernoulli Naive Bayes** models.
   - Used **GridSearchCV** for hyperparameter optimization and evaluated using **Stratified K-Fold Cross-Validation**.

3. **Model Evaluation**:
   - Compared models using **accuracy**, **F1-score**, and **ROC-AUC** metrics.
   - Performed **cross-validation** with **Stratified K-Fold** to ensure model stability.

4. **Prediction**:
   - Generated predictions on the test set and saved the results in a CSV file.

## **Files**
- **Customer_Loan_Prediction_Train.csv**: The training dataset used to train the model.
- **Customer_Loan_Prediction_Test.csv**: The test dataset used for making predictions.
- **Customer_Loan_Prediction_Test_Final.csv**: The output file containing predictions for the test dataset.

## **Technologies Used**:
- **Python** (pandas, scikit-learn)
- **Machine Learning** (Logistic Regression, Naive Bayes, Bernoulli Naive Bayes)
- **Hyperparameter Tuning** (GridSearchCV)
- **Cross-Validation** (Stratified K-Fold)
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## **Conclusion**

This project demonstrates the power of **machine learning** and **feature engineering** for predicting loan defaults in the financial industry. The optimized **Bernoulli Naive Bayes** model, coupled with **cross-validation** and **hyperparameter tuning**, provides an effective solution for forecasting loan defaults. 

By evaluating the model with multiple metrics and optimizing its parameters, we ensure that the model performs well in real-world scenarios, offering reliable predictions that financial institutions can use to assess loan approval risk.
