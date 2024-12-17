# **Customer Loan Prediction Project README**

### **Objective**  
In the realm of financial services, the ability to accurately predict loan defaults is of paramount importance. This project addresses this challenge by leveraging machine learning techniques to forecast whether a customer will default on a loan, based on various factors such as credit history, income, and demographic information.

---

### **Dataset**  

The project utilizes two key datasets:  

- **Customer_Loan_Prediction_Train.csv**:  
   This dataset comprises **814** meticulously curated customer records, encompassing crucial attributes such as gender, marital status, income details, loan parameters, credit history, and property area.  

- **Customer_Loan_Prediction_Test.csv**:  
   Serving as the evaluation ground, this dataset contains **367** records mirroring the structure of the training set but without the target variable (`Loan_Status`).

---

### **Workflow and Steps**  

1. **Data Preprocessing**:
   - Dropped unnecessary columns like `Loan_ID`.
   - Handled missing values by imputing the **mode** for categorical features and applied **Min-Max Scaling** for numerical features such as `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount`.
   - Converted categorical features into numerical form using **One-Hot Encoding**.

2. **Feature Scaling**:  
   - Applied **Min-Max Scaling** to ensure numerical features were scaled between 0 and 1, improving model performance.

3. **Model Selection and Training**:
   - Trained two machine learning models:  
     - **Logistic Regression**: Used as the baseline model.  
     - **Bernoulli Naive Bayes**: Selected as the final model after hyperparameter tuning.

4. **Hyperparameter Tuning**:
   - Used **GridSearchCV** to optimize the hyperparameters for the **Bernoulli Naive Bayes** model.  
   - The best parameters (`alpha` and `binarize`) were selected after 5-Fold Cross-Validation.

5. **Cross-Validation**:
   - To validate model robustness, **5-Fold Cross-Validation** was applied to the Bernoulli Naive Bayes model, yielding an improved accuracy of **80.78%**.

6. **Model Evaluation**:  
   - Compared the performance of the Logistic Regression and Bernoulli Naive Bayes models.  
   - Bernoulli Naive Bayes outperformed Logistic Regression with a final **cross-validated accuracy** of **80.78%**.

7. **Prediction on Test Dataset**:
   - The optimized Bernoulli Naive Bayes model was used to predict loan statuses for the test dataset.  
   - The predictions were saved in **`Customer_Loan_Prediction_Test_Final.csv`**.

---

### **Algorithm and Model Evaluation**

The project focused on the following models:  

- **Logistic Regression**: Used as a baseline for comparison.  
- **Bernoulli Naive Bayes**: Final model selected for its strong performance on binary/categorical data.  
   - Achieved an accuracy of **80.78%** using 5-Fold Cross-Validation, demonstrating its robustness.

---

### **Results**  

- Predictions were made on the test dataset (`Customer_Loan_Prediction_Test.csv`) using the optimized Bernoulli Naive Bayes model.  
- The final predictions were saved in a new file:  
   **`Customer_Loan_Prediction_Test_Final.csv`**.

---

### **Analysis**  

- **Feature Engineering**:  
   - Converted categorical variables into numerical form using One-Hot Encoding.  
   - Scaled numerical features using Min-Max Scaling to bring them within the same range.  

- **Model Selection**:  
   - Evaluated multiple models and finalized **Bernoulli Naive Bayes** as the best-performing model.  

- **Cross-Validation**:  
   - Validated model performance using **5-Fold Cross-Validation**, achieving an accuracy of **80.78%**.

---

### **Technologies Used**  
- **Programming Language**: Python  
- **Libraries**: pandas, scikit-learn, GridSearchCV  
- **Machine Learning Models**: Logistic Regression, Bernoulli Naive Bayes  
- **Techniques**: Data Preprocessing, Feature Scaling, Hyperparameter Tuning, Cross-Validation  

---

### **Conclusion**  

This project demonstrates a systematic approach to building a machine learning model for **loan default prediction**. Key achievements include:  

- Successfully trained and optimized a **Bernoulli Naive Bayes** model.  
- Achieved a **Cross-Validation Accuracy** of **80.78%**, validating the model's stability.  
- Delivered actionable predictions for loan approval using the test dataset.

---

### **Files Included**  

1. **Customer_Loan_Prediction_Train.csv**: Training dataset for model development.  
2. **Customer_Loan_Prediction_Test.csv**: Test dataset for predictions.  
3. **Customer_Loan_Prediction_Test_Final.csv**: Final output containing predicted loan statuses.  
4. **Customer_Loan_Prediction.ipynb**: The notebook containing the full implementation and code.
