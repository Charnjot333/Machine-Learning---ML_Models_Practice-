# 🚗 Used Car Price Prediction — Decision Tree Regressor

## 📋 Project Overview
This project predicts the **selling price of used cars** based on various features such as car model, year, mileage, fuel type, and transmission.  
The goal is to build a **machine learning model** that can accurately estimate used car prices and improve performance using **hyperparameter tuning**.

---

## 🧠 Machine Learning Model
- **Algorithm Used:** Decision Tree Regressor  
- **Objective:** Predict continuous target variable — `Selling Price`

---

## 🧹 Steps Involved

### 1. Data Preprocessing
- Loaded and explored dataset.
- Handled missing values and duplicates.
- Encoded categorical variables using Label Encoding / OneHotEncoding.
- Split the dataset into **train** and **test** sets (80%-20%).
- Applied **feature scaling** to normalize numeric columns.

### 2. Model Training
Trained a **Decision Tree Regressor** on the processed data:
```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
**### 3. Hyperparameter Tuning**

To improve model performance and avoid overfitting, **RandomizedSearchCV** was used to find the best combination of hyperparameters for the Decision Tree Regressor.

**#### 🔧 Parameter Grid**
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'max_depth': [None, 3, 5, 8, 12, 15, 20],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
}
###🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / VS Code



