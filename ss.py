import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
app= FastAPI()
@app.get("/data/")
def lungcancer(age:int,gender:int,family_history:int,smoking_status:float,bmi:float,cholesterol_level:int,hypertension:int,asthma:int,cirrhosis:int,other_cancer:int,treatment_type:int,cancer_stage:int):
    LR=LinearRegression()
    data_df=pd.read_csv('Lung Cancer.csv')
    x=data_df.drop('survived',axis=1)
    y=data_df['survived']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    LR.fit(x_train,y_train)
    y_pred=LR.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)

    'print(accuracy*100)'
    d={'age':[age],'gender':[gender],'family_history':[family_history],'smoking_status':[smoking_status],'bmi':[bmi],'cholesterol_level':[cholesterol_level],'hypertension':[hypertension],'asthma':[asthma],'cirrhosis':[cirrhosis],'other_cancer':[other_cancer],'treatment_type':[treatment_type],'cancer_stage':[cancer_stage]}
    df=pd.DataFrame(d)
    yp = LR.predict(df)
    if yp==1:
        return {'prediction':'patient dead'}
    else:
        return {'prediction':'patient survived'}






