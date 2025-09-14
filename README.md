# 🫀 Heart Disease Prediction & Analysis  
**Comprehensive Machine Learning Pipeline on Heart Disease UCI Dataset**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-orange.svg)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-yellow.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-brightgreen.svg)  

---

## Project Overview  
This project provides a complete **Machine Learning pipeline** for analyzing, predicting, and visualizing heart disease risks using the **UCI Heart Disease dataset**. It covers data preprocessing, dimensionality reduction (PCA), supervised & unsupervised models, hyperparameter tuning, and optional web deployment using Streamlit & Ngrok. 

---

## Table of Contents
- [Project Objectives](#project-objectives)  
- [Dataset](#dataset)
- [Tools & Libraries](#tools--libraries)
- [Project Structure](#project-structure)  
- [How to Run](#how-to-run)
  - [Clone the Repository](#clone-the-repository)  
  - [Create the Virtual Environment](#create-the-virtual-environment)  
  - [Activate the Virtual Environment](#activate-the-virtual-environment)  
  - [Install Dependencies](#install-dependencies)  
  - [Run Jupyter Notebooks](#run-jupyter-notebooks)  
  - [Run the Streamlit Web App](#run-the-streamlit-web-app)  
  - [Deploy using Ngrok](#deploy-using-ngrok)  
- [Pipeline Workflow](#pipeline-workflow)  
- [Results & Deliverables](#results--deliverables)
---

## Project Objectives  
- Perform **Data Cleaning & Preprocessing** (missing values, encoding, scaling).  
- Apply **Dimensionality Reduction** using PCA.  
- Implement **Feature Selection** using Random Forest, RFE, Chi-Square.
- Train **Supervised Models**:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
- Apply **Unsupervised Learning**:
  - K-Means Clustering  
  - Hierarchical Clustering  
- Optimize models using **GridSearchCV** & **RandomizedSearchCV**.  
- Deploy a **Streamlit Web App** and use **Ngrok** for public access.  

---

## Dataset
- **Name:** [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
- **Description:** Predict the presence or absence of heart disease based on clinical parameters.  

---

## Tools & Libraries  
- **Python** – Main programming language  
- **Pandas, NumPy** – Data Handling  
- **Matplotlib, Seaborn** – Visualization  
- **Scikit-learn** – Machine Learning Models & PCA  
- **Streamlit** – Interactive Web App 
- **Ngrok** – Public URL for Deployment 

---

## Project Structure
```
Heart_Disease_Project/
│── data/
│   ├── heart_disease.csv
│── notebooks/
│   ├── 00_data_collecting.ipynb
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│── models/
│   ├── final_model.pkl
│── ui/
│   ├── app.py (Streamlit UI)
│── deployment/
│   ├── ngrok_setup.txt
│── results/
│   ├── evaluation_metrics.txt
│── requirements.txt
│── README.md
│── .gitignore
```

---

## How to Run  

### Clone the Repository
```
git clone https://github.com/basmala-ayman/Heart-Disease.git
cd Heart-Disease
```

### Create the Virtual Environment
```
python3 -m venv venv
```

### Activate the virtual environment
```
# On macOS/Linux:
source venv/bin/activate
```
```
# On Windows:
venv\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Run Jupyter Notebooks
```
jupyter notebook
```

### Run the Streamlit Web App
```
streamlit run ui/app.py
```

### Deploy using Ngrok
Read instructions in `deployment/ngrok_setup.txt`.  

---

## Pipeline Workflow  
1. **Data Preprocessing & Cleaning** – Handle missing values, encoding, scaling  
2. **PCA Analysis** – Dimensionality Reduction  
3. **Feature Selection** – Random Forest, RFE, Chi-Square
4. **Model Training** – Logistic Regression, Decision Tree, Random Forest, SVM  
5. **Evaluation** – Accuracy, Precision, Recall, F1, ROC-AUC  
6. **Clustering** – K-Means & Hierarchical Clustering  
7. **Hyperparameter Tuning** – GridSearchCV, RandomizedSearchCV  
8. **Deployment (Bonus)** – Streamlit & Ngrok  

---

## Results & Deliverables  
- Cleaned Dataset  
- PCA & Feature Selection Results  
- Trained Models with Evaluation Metrics  
- Optimized Model Saved as `.pkl`  
- Interactive Streamlit UI
- Ngrok Public Access Link

---
