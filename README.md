# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A machine learning project to classify **Iris flower species** based on petal and sepal dimensions using the **K-Nearest Neighbors (KNN)** algorithm. The project includes data analysis, preprocessing, model training, evaluation, and predictions with clear visualizations.

---

## 📂 Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Size**: 150 samples × 5 columns
- **Features**:
  - `sepal_length`
  - `sepal_width`
  - `petal_length`
  - `petal_width`
  - `species` (target: setosa, versicolor, virginica)

---

## 🔧 Project Workflow

### 1. 🗃️ Load and Inspect Dataset
- Load the CSV using `pandas`
- Display structure and statistics

### 2. 🧼 Data Cleaning
- Check for missing values and duplicates
- Drop duplicate rows if found

### 3. 📊 Exploratory Data Analysis (EDA)
- Pairplot visualization
- Correlation heatmap
- Boxplots to inspect outliers

### 4. 🔠 Label Encoding
- Convert categorical species labels into numerical form

### 5. ✂️ Train-Test Split
- Split dataset into training (80%) and testing (20%) sets

### 6. 🧠 Model Training
- Use `KNeighborsClassifier` with `k=3`
- Train the model on training data

### 7. 📈 Model Evaluation
- Evaluate with accuracy score and classification report
- Visualize confusion matrix

### 8. 🔮 New Sample Prediction
- Predict species for a new flower input
- Visualize input features and predicted label

---

## 📸 Visualization Samples

| Pairplot | Correlation Heatmap | Input Feature Bar Chart |
|----------|---------------------|--------------------------|
| ![pairplot](https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_dataset_scatterplot.svg) | ![heatmap](https://seaborn.pydata.org/_images/seaborn-heatmap-1.png) | *Generated from matplotlib in code* |

> *Note: Replace placeholders with actual images/screenshots from your outputs if desired.*

---

## 🛠 Tech Stack

- **Programming Language**: Python 3.8+
- **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`

---

## ✅ Model Performance

| Metric        | Value |
|---------------|-------|
| Accuracy      | 100%  |
| Precision     | 1.00  |
| Recall        | 1.00  |
| F1-Score      | 1.00  |

> KNN achieved perfect accuracy on the test set (due to the simplicity and cleanliness of the Iris dataset).

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/iris-knn-classification.git
cd iris-knn-classification
