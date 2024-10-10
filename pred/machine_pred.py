import pandas as pd
df=pd.read_csv(r"C:\Users\Mohammad Fasi Ahmed\Desktop\machine learning\Predictive_Maintenance_Dataset.csv")
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])
df = df.drop(columns=['Product ID'])
x=df.drop(["Machine failure","UDI"],axis=1)
y=df["Machine failure"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train_scaled,y_train)
y_pred=reg.predict(x_test_scaled)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_pred,y_test)
cr=classification_report(y_pred,y_test)
sc=accuracy_score(y_pred,y_test)
print(cr)
print(cm)
print(sc)
from sklearn.tree import DecisionTreeClassifier
tr=DecisionTreeClassifier()
tr.fit(x_train_scaled,y_train)
y_pred1=tr.predict(x_test_scaled)
cr1=classification_report(y_test,y_pred1)
sc1=accuracy_score(y_test,y_pred1)
print(cr1)
print(sc1)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train_scaled,y_train)
y_pred2=svc.predict(x_test_scaled)
cr2=classification_report(y_pred2,y_test)
sc2=accuracy_score(y_pred2,y_test)
print(cr2)
print(sc2)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train_scaled,y_train)
y_pred3=rfc.predict(x_test_scaled)
cr3=classification_report(y_pred3,y_test)
sc3=accuracy_score(y_pred3,y_test)
print(cr3)
print(sc3)
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(x_train_scaled,y_train)
y_pred4=knc.predict(x_test_scaled)
cr4=classification_report(y_test,y_pred4)
sc4=accuracy_score(y_test,y_pred4)
print(cr4)
print(sc4)
import seaborn as sns
import matplotlib.pyplot as plt
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
for feature in features:
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train_scaled,y_train)
y_pred5=gnb.predict(x_test_scaled)
from sklearn.metrics import classification_report,accuracy_score
cr5=classification_report(y_test,y_pred5)
sc5=accuracy_score(y_test,y_pred5)
print(cr5)
print(sc5)
model_input = [[1, 298.1, 308.5, 1800, 35.0, 130, 0, 0, 0, 0, 0]]
model_input_scaled = scaler.transform(model_input)
model_output = tr.predict(model_input_scaled)

if model_output==0:
    print("no failure")
else:
    print("failure")
import pickle
filename="machine_pred.pkl"
with open(filename,"wb") as file:
    pickle.dump(tr,file)
print("file has been saved successfully")
import os 
os.getcwd()