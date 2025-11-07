import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import accuracy_score




# Step 1: Load your dataset
data = pd.read_csv("Crop_recommendation.csv")

X=data.drop('label',axis=1)
y=data['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=42)
model=RandomForestClassifier()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print("Accuracy: ",accuracy_score(y_test,predictions))

import joblib
joblib.dump(model,"crop_app.pkl")
