
# Import dependencies
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("customer_churn_business_dataset.csv")

# Drop unnecessary columns
data = df.drop(columns=[
    'features_used','usage_growth_rate','last_login_days_ago','monthly_fee',
    'total_revenue','payment_method','payment_failures','discount_applied',
    'price_increase_last_3m','support_tickets','avg_resolution_time',
    'complaint_type','csat_score','escalations','email_open_rate',
    'marketing_click_rate','nps_score','survey_response','referral_count'
])

#  Separate categorical and numerical columns
x = data.drop('churn', axis=1)
y = data['churn']

# Numeric columns only
num_list = x.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Categorical columns
cat_list = x.select_dtypes(include=['object']).columns.tolist()

# Exclude contract_type from one-hot, it will be ordinal
other_cat = [col for col in cat_list if col != 'contract_type']

#  ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_list),
        ('contract', OrdinalEncoder(categories=[['Monthly', 'Quarterly', 'Yearly']]), ['contract_type']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), other_cat)
    ],
    remainder='passthrough'
)

#  Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#  Pipeline
pipeline = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))  # handles class imbalance
    ]
)

#  Fit the model
pipeline.fit(x_train, y_train)

# Save the trained pipeline as pickle
with open('churn_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("✅ Model trained and saved as 'churn_pipeline.pkl'")
