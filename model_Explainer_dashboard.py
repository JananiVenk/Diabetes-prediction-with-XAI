import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from explainerdashboard import ClassifierExplainer,ExplainerDashboard

from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("diabetes.csv")

feature_name=list(data.columns)
feature_name.remove("Outcome")

def drop_ouliers_iqr(data,feature_name):
    q1=data[feature_name].quantile(0.25)
    q3=data[feature_name].quantile(0.75)
    iqr=q3-q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    upper_array = np.where(data[feature_name] >= upper)[0]
    lower_array = np.where(data[feature_name] <= lower)[0]
    data=data.drop(upper_array).reset_index(drop=True)
    data=data.drop(lower_array).reset_index(drop=True)
    return data

for feat in feature_name:
    data=drop_ouliers_iqr(data,feat)

x=data[feature_name]
y=data['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

model=RandomForestClassifier(n_estimators=1000)
model.fit(x_train,y_train)
preds=model.predict(x_test)
print("Accuracy score:",accuracy_score(y_test,preds)*100)

explainer=ClassifierExplainer(model,x_test,y_test)
dashboard=ExplainerDashboard(explainer)
dashboard.run()







