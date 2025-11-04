# 🧠 Purchase Prediction using Gaussian Naive Bayes

This project predicts whether a person will purchase a product based on their **Age** and **Estimated Salary** using the **Gaussian Naive Bayes** classification algorithm.  

---

## 📘 Project Overview

The goal of this project is to build a **machine learning model** that classifies customers into two categories:
- **0 → Not Purchased**
- **1 → Purchased**

The algorithm used is **Gaussian Naive Bayes (GNB)** — a probabilistic classifier based on **Bayes’ Theorem** with the assumption that features follow a **Gaussian (normal) distribution**.

---

## 🧩 Dataset

**File:** `purchase_logistic.csv`

| Feature | Description |
|----------|--------------|
| Age | Age of the customer |
| EstimatedSalary | Estimated salary of the customer |
| Purchased | Target variable (0 = Not Purchased, 1 = Purchased) |
| Gender | (If present) Used for analysis and label encoding |

---

## ⚙️ Tech Stack

- **Python**
- **pandas** → Data loading and preprocessing  
- **NumPy** → Numerical computations  
- **Matplotlib / Seaborn** → Data visualization  
- **scikit-learn (sklearn)** → ML model building and evaluation  

---

## 🚀 Steps Performed

### 1️⃣ Data Preprocessing
- Loaded dataset using `pandas`
- Checked for missing values and data types
- Applied **Label Encoding** for categorical columns (e.g., Gender, Purchased)
- Scaled features (`Age`, `EstimatedSalary`) using `StandardScaler` from sklearn

### 2️⃣ Model Building
- Split the data into training and testing sets (75% - 25%)
- Trained a **Gaussian Naive Bayes (GaussianNB)** model
- Predicted the target on the test set

### 3️⃣ Model Evaluation
- Generated **Confusion Matrix**
- Displayed results using **Seaborn Heatmap**
- Printed **Classification Report** (Accuracy, Precision, Recall, F1-score)

---

## 📊 Results Visualization

Confusion Matrix (via Seaborn):
```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Not Purchased', 'Purchased'],
            yticklabels=['Not Purchased', 'Purchased'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Gaussian Naive Bayes)')
plt.show()
```
## 👩‍💻 Author

**Charanjot Kaur**  
🎓 MCA Student | 💻 Machine Learning & Data Science Enthusiast  

- 📫 **Email:** [charanjotkaur1@gmail.com]  
- 🌐 **GitHub:** [https://github.com/Charnjot333](https://github.com/Charnjot333)  
- 🔗 **LinkedIn:** [https://linkedin.com/in/charanjot-kaur-arora](https://linkedin.com/in/charanjot-kaur-arora)  

> *“Learning by building — one project at a time.”*
