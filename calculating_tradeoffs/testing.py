import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load actual data
data_path = '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/GRU1_combined.csv'
data = pd.read_csv(data_path)

# Ensure data is loaded correctly
logger.info(f"Data loaded with shape: {data.shape}")
logger.info(f"Columns: {data.columns.tolist()}")

# Define target and features
target = ['career aspirations', 'future topics', 'ethnoracial group', 'gender', 'international status']
features = [col for col in data.columns if col != target]

# Preprocess data
X = data[features]
y = data[target]

# Log initial state
logger.info(f"Initial feature names: {X.columns.tolist()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure feature names are retained
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

# Log after split
logger.info(f"Feature names after split (train): {X_train.columns.tolist()}")
logger.info(f"Feature names after split (test): {X_test.columns.tolist()}")

# Define categorical and numerical columns
categorical_cols = ['ethnoracial group', 'international status','gender']
numerical_cols = ['learning style', 'gpa', 'student semester', 'major', 'previous courses', 'course types', 'course subjects', 'subjects of interest', 'extracurricular activities', 'career aspirations', 'future topics']

# Ensure columns exist in the data
missing_categorical_cols = [col for col in categorical_cols if col not in X_train.columns]
missing_numerical_cols = [col for col in numerical_cols if col not in X_train.columns]
assert not missing_categorical_cols, f"Missing categorical columns: {missing_categorical_cols}"
assert not missing_numerical_cols, f"Missing numerical columns: {missing_numerical_cols}"

# Define preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

# Fit and transform the data using the preprocessor
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after preprocessing
num_col_names = numerical_cols
cat_col_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
all_col_names = num_col_names + cat_col_names

# Convert the resulting numpy array to a DataFrame
X_train_processed = pd.DataFrame(X_train_processed, columns=all_col_names)
X_test_processed = pd.DataFrame(X_test_processed, columns=all_col_names)

# Log after preprocessing
logger.info(f"Feature names after preprocessing (train): {X_train_processed.columns.tolist()}")
logger.info(f"Feature names after preprocessing (test): {X_test_processed.columns.tolist()}")

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_processed, y_train)

# Log feature names before prediction
logger.info(f"Feature names before prediction (test): {X_test_processed.columns.tolist()}")

# Make predictions
y_pred = model.predict(X_test_processed)

# Log predictions
logger.info(f"Predictions: {y_pred}")

# Evaluate model
accuracy = model.score(X_test_processed, y_test)
logger.info(f"Model accuracy: {accuracy}")
