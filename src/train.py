import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Load the cleaned data
df = pd.read_csv('C:\\Users\\Surface\\OneDrive\\Documentos\\GitHub\\Titanic-survival-rate\\data\\titanic_cleaned_new.csv')

# Features and target variable
X = df.drop(['Survived'], axis=1)   
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)
print(report)

# Save the trained model
# import joblib
# import os
# os.makedirs('model', exist_ok=True)
# joblib.dump(clf, 'model/random_forest_model.joblib')
# # Save feature columns
# joblib.dump(X.columns.tolist(), 'model/feature_columns.joblib')

# save with pickle
# ...existing code...

# Efficient saving function
import os
import joblib

def save_model_and_features(model, features, folder='model'):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, os.path.join(folder, 'random_forest_new_model.joblib'))
    joblib.dump(features, os.path.join(folder, 'feature_new_columns.joblib'))

save_model_and_features(clf, X.columns.tolist())