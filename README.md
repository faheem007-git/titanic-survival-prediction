# 🚢 Titanic Survival Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning. The model is trained on passenger information such as age, gender, ticket class, fare, and embarkation port.

The objective is to build a classification model that can accurately predict passenger survival.

---

## 🎯 Problem Statement

The Titanic disaster is one of the most well-known tragedies in history.

Using passenger information, we aim to answer the question:

**"What types of passengers were more likely to survive?"**

This project uses **Logistic Regression**, a supervised machine learning classification algorithm, to predict survival outcomes.

---

## 📊 Dataset

**Source:** Kaggle - Titanic: Machine Learning from Disaster

### Target Variable

```python
Survived
```

### Target Values

```text
0 = Did Not Survive
1 = Survived
```

### Features Used

```python
Pclass      # Passenger Class
Sex         # Gender
Age         # Passenger Age
Fare        # Ticket Fare
Embarked    # Port of Embarkation
```

---

## 🛠 Technologies Used

### Programming Language

- Python

### Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle
```

### Tools

- VS Code
- Jupyter Notebook
- Git
- GitHub

---

## 📚 Machine Learning Concepts Used

### Data Cleaning

- Handling missing values
- Removing unnecessary columns

### Feature Engineering

- Converting categorical values into numerical values
- Preparing data for machine learning

### Classification

- Logistic Regression

### Model Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 🔄 Project Workflow

### 1. Import Libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

---

### 2. Load Dataset

```python
df = pd.read_csv("train.csv")
```

---

### 3. Explore Dataset

```python
df.head()

df.info()

df.describe()
```

---

### 4. Check Missing Values

```python
df.isnull().sum()
```

---

### 5. Data Cleaning

Fill missing Age values:

```python
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)
```

Fill missing Embarked values:

```python
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)
```

Remove Cabin column:

```python
df.drop(
    columns=["Cabin"],
    inplace=True
)
```

---

### 6. Convert Categorical Data

Convert Gender:

```python
df["Sex"] = df["Sex"].map(
    {
        "male":0,
        "female":1
    }
)
```

Convert Embarked:

```python
df["Embarked"] = df["Embarked"].map(
    {
        "S":0,
        "C":1,
        "Q":2
    }
)
```

---

### 7. Feature Selection

```python
X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked"
    ]
]

y = df["Survived"]
```

---

### 8. Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 9. Train Model

```python
model = LogisticRegression()

model.fit(
    X_train,
    y_train
)
```

---

### 10. Make Predictions

```python
predictions = model.predict(
    X_test
)
```

---

### 11. Evaluate Model

#### Accuracy Score

```python
accuracy = accuracy_score(
    y_test,
    predictions
)

print(accuracy)
```

#### Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)
```

#### Classification Report

```python
print(
    classification_report(
        y_test,
        predictions
    )
)
```

---

### 12. Save Model

```python
import pickle

pickle.dump(
    model,
    open(
        "titanic_model.pkl",
        "wb"
    )
)
```

---

## 📈 Results

### Accuracy

```text
Approximately 78% - 85%
```

The exact accuracy depends on preprocessing and feature selection.

### Evaluation Metrics

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📊 Visualization

Passenger Survival Distribution

```python
sns.countplot(
    x="Survived",
    data=df
)

plt.show()
```

---

## 🧪 Sample Prediction

### Input

```python
new_passenger = pd.DataFrame(
    [[1,1,25,100,1]],
    columns=[
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked"
    ]
)
```

### Predict

```python
prediction = model.predict(
    new_passenger
)

print(prediction)
```

### Output

```text
[1]
```

Meaning:

```text
Passenger Likely Survives
```

---

## 📂 Project Structure

```text
titanic-survival-prediction/
│
├── train.csv
├── titanic.ipynb
├── titanic_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── images/
```

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
```

### Move Into Project Directory

```bash
cd titanic-survival-prediction
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook
```

or

```bash
python titanic.py
```

---

## 🎓 Learning Outcomes

Through this project, I learned:

- Data Cleaning and Preprocessing
- Handling Missing Values
- Categorical Encoding
- Feature Selection
- Logistic Regression
- Classification Problems
- Model Evaluation Techniques
- Machine Learning Workflow
- Model Serialization using Pickle
- Git and GitHub

---

## 🔮 Future Improvements

- Random Forest Classifier
- XGBoost Classifier
- Hyperparameter Tuning
- Feature Engineering
- Flask Deployment
- Streamlit Deployment
- Interactive Dashboard

---

## 👨‍💻 Author

**Faheem Ali Mirza**

GitHub:
https://github.com/faheem007-git

LinkedIn:
https://www.linkedin.com/in/faheem-ali-mirza-a41786344

---

## ⭐ If you found this project useful, please consider giving it a star.
