

# Customer Churn Prediction Project

Predicting **customer churn** is a critical business problem. Retaining customers costs far less than acquiring new ones. This project demonstrates a **complete end-to-end Machine Learning workflow**, including data preprocessing, modeling, evaluation, and deployment — making it a **portfolio-ready showcase**.

---

## Problem Statement

Customer churn occurs when a customer **leaves a platform or service**, which negatively impacts:

- Revenue  
- Growth  
- Customer retention  

Predicting churn allows businesses to:

- Identify **customers at risk**  
- Implement **retention strategies**  
- Make **data-driven decisions**  

This project solves the problem by building a **predictive model** to estimate churn probability for each customer.

---

## Skills & Concepts Demonstrated

| Skill / Concept            | Description |
|----------------------------|------------|
| Data Exploration (EDA)     | Analyze dataset, visualize churn patterns, handle missing values |
| Preprocessing Pipelines    | ColumnTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder |
| Modeling                   | Logistic Regression in a pipeline |
| Evaluation                 | Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix |
| Deployment                 | Streamlit interactive web app |
| Tools                      | pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit |
| Portfolio Skills           | End-to-end ML workflow, preprocessing strategies, deployment, business problem-solving |

---

## Project Structure

```

customer-churn-prediction/
├── README.md                     # Project overview & portfolio showcase
├── requirements.txt              # Dependencies
├── data/                         # Sample dataset
│   └── customer_churn_business_dataset.csv
├── notebook/                     # Jupyter Notebook for exploration & modeling
│   └── churn_analysis.ipynb
├── model/                        # Saved trained model
│   └── churn_pipeline.pkl
├── backend/                      # Scripts for training & pickle generation
│   └── main_model_code.py
├── frontend/                     # Streamlit app for predictions
│   └── app.py

````

---

## End-to-End ML Workflow

1. **Data Collection & Loading**  
   - Load customer dataset (`.csv`) into pandas DataFrame.  
   - Inspect data types, missing values, and statistics.

2. **Exploratory Data Analysis (EDA)**  
   - Visualize churn vs non-churn patterns.  
   - Identify trends, correlations, and anomalies.

3. **Data Preprocessing & Feature Engineering**  
   - Handle missing values and outliers.  
   - Encode categorical variables using OneHotEncoder / OrdinalEncoder.  
   - Scale numerical features using StandardScaler.  
   - Use `ColumnTransformer` and `Pipeline` for reproducibility.

4. **Model Training**  
   - Train Logistic Regression classifier.  
   - Use pipeline integration to combine preprocessing and modeling.  
   - Apply cross-validation.

5. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
   - Visualizations: Confusion Matrix, ROC Curve.  
   - Interpret feature importance.

6. **Model Saving & Deployment**  
   - Save trained pipeline with `pickle`.  
   - Deploy with **Streamlit**:  
     - Interactive web app for customer input.  
     - Real-time churn probability display.

---

## How to Run

1. Clone repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit app:

```bash
streamlit run frontend/app.py
```

4. Open browser at `http://localhost:8501` to interact with the app.

---

## Highlights

* End-to-end ML workflow: Data → Model → Deployment
* Problem-solving skills: Real business problem (churn prediction)
* Technical expertise: Pipelines, preprocessing, evaluation, deployment
* Business impact: Helps companies retain customers & reduce revenue loss



