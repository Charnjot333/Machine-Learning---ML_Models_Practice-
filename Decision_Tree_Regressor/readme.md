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
- 
### 2. ⚙️ Model Training
Trained a **Decision Tree Regressor** on the processed data:

```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
```
###3. 🔧 Hyperparameter Tuning
To improve model performance and prevent overfitting, RandomizedSearchCV was used to find the optimal combination of hyperparameters for the Decision Tree Regressor.
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'max_depth': [None, 3, 5, 8, 12, 15, 20],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
}
```
## 🎯 Best Parameters Found
```python
{
 'min_samples_split': 20,
 'min_samples_leaf': 10,
 'max_features': None,
 'max_depth': 12,
 'criterion': 'squared_error'
}
```
---
### 📊 Results & Insights

* The model achieved an R² score of 0.69 on the test set, indicating a decent level of accuracy.

* Hyperparameter tuning significantly improved performance and reduced overfitting.

* Model errors (MAE and RMSE) are reasonable considering the variability in real-world car prices.
### 3. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / VS Code
## 👩‍💻 Author

**Charanjot Kaur**  
🎓 MCA Student | 💻 Machine Learning & Data Science Enthusiast  

- 📫 **Email:** [charanjotkaur1@gmail.com]  
- 🌐 **GitHub:** [https://github.com/Charnjot333](https://github.com/Charnjot333)  
- 🔗 **LinkedIn:** [https://linkedin.com/in/charanjot-kaur-arora](https://linkedin.com/in/charanjot-kaur-arora)  

> *“Learning by building — one project at a time.”*



