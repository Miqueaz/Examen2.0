import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)



df = pd.read_csv("diabetes.csv")

cols_with_zeros_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[cols_with_zeros_as_missing] = df[cols_with_zeros_as_missing].replace(0, np.nan)


X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    
    random_state=42,  
    stratify=y        
)



preprocessing_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())                  
])

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessing_pipeline),
    ('classifier', RandomForestClassifier(random_state=42)) 
])



final_pipeline.fit(X_train, y_train)



y_pred = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1]


acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_proba)


model_filename = 'diabetes_pipeline.joblib'
joblib.dump(final_pipeline, model_filename)

print(f"\n¡Éxito! El pipeline entrenado ha sido guardado como '{model_filename}'.")