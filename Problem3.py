import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv('/Users/guangjingzhu/Desktop/statistic learning/HW8/hepatitis.csv')

# Data Processing
# (a) Drop the ID column
data = data.drop('ID', axis=1)

# (b) Check and impute null values
null_columns = data.columns[data.isnull().any()]

for col in null_columns:
    mean_value = data[col].mean()
    data[col].fillna(mean_value, inplace=True)

data.replace('?', np.nan, inplace=True)
# Convert columns to appropriate data types
data = data.apply(pd.to_numeric, errors='ignore')

# Calculate mean for each column
mean_values = data.mean()

# Replace NaN values with the mean for each column
data.fillna(mean_values, inplace=True)

# Separate features and target variable
X = data.drop('histology', axis=1)
y = data['histology']

# Split the data into train and test sets (90:10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model Building
# SVM model with default parameters
svm_default = SVC()
svm_default.fit(X_train, y_train)

# Report performance on test data with default parameters
y_pred_default = svm_default.predict(X_test)
print("SVM with Default Parameters:")
print("Accuracy:", accuracy_score(y_test, y_pred_default))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default))

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
svm_grid = SVC()
grid_search = GridSearchCV(svm_grid, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from hyperparameter tuning
best_svm = grid_search.best_estimator_

# Report performance on test data with the best model
y_pred_best = best_svm.predict(X_test)
print("\nSVM with Tuned Hyperparameters:")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))
