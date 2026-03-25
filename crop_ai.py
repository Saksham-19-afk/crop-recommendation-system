import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("Crop_recommendation.csv") #loading the dataset

X=data.drop('label',axis=1) #feature extraction x= features, y = label/target
y=data['label']

#train test split 80 to 20 ratio
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=42)

#randomforest model with 100 decision trees and max depth of 6
model=RandomForestClassifier(n_estimators=50,max_depth=6)

#training the data
model.fit(X_train,y_train)

#predicting the output
predictions = model.predict(X_test)

#accurscy
print("Accuracy:  ",accuracy_score(y_test,predictions) * 100, " %") 


joblib.dump(model,"crop_app.pkl")


