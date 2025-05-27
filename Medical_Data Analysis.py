# -*- coding: utf-8 -*-
"""의료 데이터분석"""

# 파일 로딩
with open('diabetes.csv', 'rb') as f:
    uploaded = {'diabetes.csv': f.read()}

import io
import pandas as pd
df = pd.read_csv(io.BytesIO(uploaded['diabetes.csv']))

# 데이터 확인
df.head(30)
df.tail(30)
df.isnull()
df.isnull().sum()

# 기본 통계
df["Outcome"].value_counts()
df["Pregnancies"].value_counts()
df["Age"].value_counts()

# 연령대와 임신 횟수별 분포
age_pregnancy_counts = df.groupby(['Age', 'Pregnancies']).size().reset_index(name='Count')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(age_pregnancy_counts)

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# Age 히스토그램
plt.figure(figsize=(4, 4))
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# 여러 변수 히스토그램
fig, ax = plt.subplots(4, 2, figsize=(12, 16))
sns.histplot(df["Pregnancies"], ax=ax[0, 0], kde=True)
sns.histplot(df["Glucose"], ax=ax[0, 1], kde=True)
sns.histplot(df["BloodPressure"], ax=ax[1, 0], kde=True)
sns.histplot(df["SkinThickness"], ax=ax[1, 1], kde=True)
sns.histplot(df["Insulin"], ax=ax[2, 0], kde=True)
sns.histplot(df["BMI"], ax=ax[2, 1], kde=True)
sns.histplot(df["DiabetesPedigreeFunction"], ax=ax[3, 0], kde=True)
sns.histplot(df["Age"], ax=ax[3, 1], kde=True)
plt.tight_layout()
plt.show()

# 히트맵
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Age > 40 비율 파이차트
data = [
    (df["Age"] > 40).value_counts(normalize=True)[True],
    (df["Age"] > 40).value_counts(normalize=True)[False]
]
categories = ["True(40up)", "False(40down)"]

plt.figure()
plt.pie(data, labels=categories)
plt.title("Age > 40")
plt.show()

# SkinThickness > 30 비율 파이차트
data = [
    (df["SkinThickness"] > 30).value_counts(normalize=True)[True],
    (df["SkinThickness"] > 30).value_counts(normalize=True)[False]
]
categories = ["True(30up)", "False(30down)"]

plt.figure()
plt.pie(data, labels=categories, autopct='%0.1f%%')
plt.title("SkinThickness > 30")
plt.show()

# 범용 파이차트 함수
def showPie(data, number):
    data_true = (data > number).value_counts(normalize=True).get(True, 0)
    data_false = (data > number).value_counts(normalize=True).get(False, 0)
    data_set = [data_true, data_false]
    categories = ["True", "False"]
    
    plt.figure()
    plt.pie(data_set, labels=categories, autopct="%0.1f%%")
    plt.title(f"{data.name} > {number}")
    plt.show()

showPie(df["Pregnancies"], 2)

# 데이터 분리
x = df.drop('Outcome', axis=1)
y = df["Outcome"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

LRModel = LogisticRegression(solver='liblinear', random_state=0)
LRModel.fit(x_train, y_train)

# 정확도
print("Logistic Regression Accuracy:", LRModel.score(x_test, y_test))

# 혼동 행렬
print("Confusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, LRModel.predict(x_test)))

# XGBoost
from xgboost import XGBClassifier
XGBModel = XGBClassifier()
XGBModel.fit(x_train, y_train)

print("XGBoost Accuracy:", XGBModel.score(x_test, y_test))
print("Confusion Matrix (XGBoost):")
print(confusion_matrix(y_test, XGBModel.predict(x_test)))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
RandomForestModel = RandomForestClassifier(n_estimators=100)
RandomForestModel.fit(x_train, y_train)

print("Random Forest Accuracy:", RandomForestModel.score(x_test, y_test))
print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, RandomForestModel.predict(x_test)))
