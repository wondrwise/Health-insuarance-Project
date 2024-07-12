# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:50:04 2024

@author: edwar
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("C:/Users/edwar/Desktop/working/Insuarance Project/Car insuarance.csv/test.csv")


print(df.shape)
print(df.info())

#
#
#
#

  # DATA CLEANING #

## Changing date columns to date time. ie- DateTimeOfAccident & DateReported



df = df.dropna()

df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'])

df['DateReported'] = pd.to_datetime(df['DateReported'])

#
#
#

#  Feature engineering

# ENCODING Gender, Marital Status, parttime/fulltime

## One hot encoding
encoded_df = pd.get_dummies(df, columns=['Gender', 'PartTimeFullTime'])
summary= encoded_df.describe(all())

## Label Encoding
### Gender

label_encoder_gender = LabelEncoder()

df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])

## ParttimeFulltime

label_encoder_part_full = LabelEncoder()

df['PartTimeFullTime'] = label_encoder_part_full.fit_transform(df['PartTimeFullTime'])

## Marital Status

label_encoder_marital_status = LabelEncoder()

df['MaritalStatus'] = label_encoder_marital_status.fit_transform(df['MaritalStatus'])

summary= df.describe()

df['LagTime'] = df['DateReported'] - df['DateTimeOfAccident']
df['LagTime'] = df['LagTime'].dt.days

### Binning Continours variables

### Binning Age

Age_counts= df['Age'].value_counts()

age_bins = [0,18,35,50, float('inf')] #Bins: 0-18, 19-35, 35-50, 51+
age_labels = ['Teenagers', 'Young-Adults', 'Adults', 'Seniors']

df['Age-Groups'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

print(df[['Age', 'Age-Groups']])

## Monthly wages

df['MonthlyWages'] = df['WeeklyWages']*4

df['Dependents'] = df['DependentChildren'] + df['DependentsOther']

#
#
#
#
#

# EDA

## Date Time Analysis

### DateTimeOfAccident

df['AccidentYear'] = df['DateTimeOfAccident'].dt.year
df['AccidentMonth'] = df['DateTimeOfAccident'].dt.month
df['AccidentDay'] = df['DateTimeOfAccident'].dt.day

### DateReported

df['ReportedYear'] = df['DateReported'].dt.year
df['ReportedMonth'] = df['DateReported'].dt.month
df['ReportedDay'] = df['DateReported'].dt.day

#
#
#

#### Histogram

##### Date Time Distribution

plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
sns.histplot(df['AccidentYear'], bins=20)
plt.title('Accident Year Distribution')

plt.subplot(2,2,2)
sns.histplot(df['AccidentMonth'], bins=12)
plt.title('Accident Month Distribution')

plt.subplot(2,2,3)
sns.histplot(df['ReportedYear'], bins=20)
plt.title('Reported Year Distribution')

plt.subplot(2,2,4)
sns.histplot(df['ReportedMonth'], bins=12)
plt.title('Reported Month Distributions')

plt.tight_layout()
plt.show()

### Lag Time Distribution

plt.figure(figsize= (20, 6))

sns.histplot(df['LagTime'])
plt.title('Lag Time Distribution')

plt.show()

### Demographic Analysis

plt.figure(figsize= (14,8))

plt.subplot(1,3,1)
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')

plt.subplot(1,3,2)
sns.countplot(x='MaritalStatus', data=df)
plt.title('Marital Status Distribution')

plt.subplot(1,3,3)
sns.countplot(x='PartTimeFullTime', data=df)
plt.title('Employment Type Distribution')

plt.tight_layout()
plt.show()

### Employment

plt.figure(figsize=(14,8))

plt.subplot(1,2,1)
sns.histplot(df['WeeklyWages'], bins=20)
plt.title('Weekly Wages Distribution')

plt.subplot(1,2,2)
sns.countplot(df['HoursWorkedPerWeek'], bins=20)
plt.title('Hours Worked Distribution')

plt.tight_layout()
plt.show()



#
#
#

#### Box plot

## Lag Time distribution

plt.figure(figsize=(10,14))
sns.boxplot(data=df['LagTime'])
plt.show()

#### Claims by Demographic

plt.figure(figsize=(14,8))

plt.subplot(1,3,1)
sns.boxplot(x='Gender', y='InitialIncurredCalimsCost', data=df)
plt.title('Claims by Gender')

plt.subplot(1,3,2)
sns.boxplot(x='MaritalStatus', y='InitialIncurredCalimsCost', data=df)
plt.title('Claims by Marital Status')

plt.subplot(1,3,3)
sns.boxplot(x='PartTimeFullTime',  y='InitialIncurredCalimsCost', data=df)
plt.title('Claims by Employment Type')  

plt.tight_layout()
plt.show()

#### Hours Worked / Week Wages

plt.figure(figsize=(14,8))

plt.subplot(1,2,1)
sns.boxenplot(data=df['HoursWorkedPerWeek'])
plt.title('Hours Worked Distribution')

plt.subplot(1,2,2)
sns.boxenplot(data=df['WeeklyWages'])
plt.title('Weekly Wage Distribution')

plt.tight_layout()
plt.show()

#

# AGE GROUPS ANALYSIS

#

#### Distribution of Age groups

age_groups_counts = df['Age-Groups'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x= age_groups_counts.index, y = age_groups_counts.values)
plt.title('Age Group Distribution')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

## Age group with initial incured claim cost

age_group_cost = df.groupby('Age-Groups')['InitialIncurredCalimsCost'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x= 'Age-Groups', y='InitialIncurredCalimsCost', data= age_group_cost)
plt.title('Avegrage Initial Incurred Claim Cost by Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Average Initial Incurred Claim Cost')
plt.show()

## AGE Groups with Weekly and Monlthly earning

age_group_wages = df.groupby('Age-Groups')[['WeeklyWages', 'MonthlyWages']].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='Age-Groups', y='WeeklyWages', data= age_group_wages)
plt.title('Average Weekly Wages by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Weekly Wages')
plt.show()

# Gender with weekly Earnings

gender_wages = df.groupby('Gender')[['WeeklyWages', 'InitialIncurredCalimsCost']].mean().reset_index()

gender_wages['Gender'] = gender_wages['Gender'].map({0: 'Female', 1: 'Male'})

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
sns.barplot(x='Gender', y='WeeklyWages', data=gender_wages)
plt.title('Average Weekly wages by Gender')

plt.subplot(1,2,2)
sns.barplot(x= 'Gender', y= 'InitialIncurredCalimsCost', data = gender_wages)
plt.title('Average incurred cost by Gender')

plt.xlabel('Gender')
plt.show()

#
#

# L A G  T I M E

## Age groups with lag time

age_group_lagtime = df.groupby('Age-Groups')['LagTime'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='Age-Groups', y='LagTime', data=age_group_lagtime)
plt.title('Average Lag Time by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average lag time (days)')
plt.show()

## Gender with lag time

gender_lagtime = df.groupby('Gender')['LagTime'].mean().reset_index()

gender_lagtime['Gender']= gender_lagtime['Gender'].map({0: 'Female', 1: 'Male'})

## Marital Status with lag time

marital_status_lagtime = df.groupby('MaritalStatus')['LagTime'].mean().reset_index()

marital_status_lagtime['MaritalStatus'] = marital_status_lagtime['MaritalStatus'].map({0: 'Married', 1: 'Single', 2: 'U'})

## Lag time by employment type

employment_type_lagtime = df.groupby('PartTimeFullTime')['LagTime'].mean().reset_index()

employment_type_lagtime['PartTimeFullTime'] = employment_type_lagtime['PartTimeFullTime'].map({0: 'FullTime', 1: 'PartTime'})

plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
sns.barplot(x='Age-Groups', y='LagTime', data=age_group_lagtime)
plt.title('Average Lag Time by Age Group')
plt.xlabel('Age Group')

plt.subplot(2,2,2)
sns.barplot(x='Gender', y='LagTime', data= gender_lagtime)
plt.title('Average Lag Time by Gender')
plt.xlabel('Gender')

plt.subplot(2,2,3)
sns.barplot(x='MaritalStatus', y= 'LagTime', data= marital_status_lagtime)
plt.title('Average Lag Time by Marital Status')
plt.xlabel('Marital Status')

plt.subplot(2,2,4)
sns.barplot(x='PartTimeFullTime', y= 'LagTime', data= employment_type_lagtime)
plt.title('Average Lag Time by Employment Type')
plt.xlabel('Employment Type')

plt.tight_layout()
plt.show()


## Lag Time vs Claim cost

plt.figure(figsize=(10, 6))
sns.scatterplot(x='LagTime', y= 'InitialIncurredCalimsCost', data=df)
plt.title('Lag Time vs Incurred Claim Cost')
plt.xlabel('Lag Time(days)')
plt.ylabel('Incurred Claim Cost')
plt.show()

#
#
#
#

# Time series analysis of the LAG TIME

#
#

df['AccidentMonthYear'] = df['DateTimeOfAccident'].dt.to_period('M') # Monthly aggregation
monthly_lag = df.groupby('AccidentMonthYear')['LagTime'].mean().reset_index()

monthly_lag['AccidentMonthYear'] = monthly_lag['AccidentMonthYear'].dt.to_timestamp()
monthly_lag.set_index('AccidentMonthYear', inplace=True)

monthly_claim_cost = df.groupby('AccidentMonthYear')['InitialIncurredCalimsCost'].mean().reset_index()


#

plt.figure(figsize=(20,6))
plt.plot(monthly_lag.index, monthly_lag['LagTime'], marker='o', linestyle='-')
plt.title('Average Lag Time Over Time')
plt.xlabel('Date')
plt.ylabel('Average Lag time')
plt.grid(True)
plt.show()


plt.figure(figsize=(14,7))
plt.subplot(2,1,1)
plt.plot(monthly_lag.index, monthly_lag['LagTime'], marker='o')
plt.title('Average Lag Time Over Time')
plt.xlabel('Date')
plt.ylabel('Lag Time (Days)')

plt.subplot(2,1,2)
plt.plot(monthly_claim_cost['AccidentMonthYear'].dt.to_timestamp(), monthly_claim_cost['InitialIncurredCalimsCost'], marker='o')
plt.title('Average Claim Cost Over Time')
plt.xlabel('Date')
plt.ylabel('Average Claim Cost')

plt.tight_layout()
plt.show()

# Decomposition of the Time series

## LAG TIME

decomposition = seasonal_decompose(monthly_lag['LagTime'], model='additive')
decomposition.plot()
plt.show()

## Monthly Claim cost

monthly_claim_cost['AccidentMonthYear'] = monthly_claim_cost['AccidentMonthYear'].dt.to_timestamp()
monthly_claim_cost.set_index('AccidentMonthYear', inplace=True)
decomposition = seasonal_decompose(monthly_claim_cost['InitialIncurredCalimsCost'], model='additive')
plt.figure(figsize=(12,6))
decomposition.plot()
plt.show()


#
#
#

# Correlation Analysis

## Correlation Matrix

columns_of_interest = [
    'DateTimeOfAccident', 'DateReported','Age','Gender','MaritalStatus','Dependents',
    'WeeklyWages', 'PartTimeFullTime', 'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialIncurredCalimsCost',
    'LagTime',
    ]

correlation_matrix = df[columns_of_interest].corr()

print(correlation_matrix)


sns.set(style='whitegrid')

# visualise correlation matrix heatmap

plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('correlation Matrix for specified columns')
plt.show()


#
#
#


# Feature Importance


## Random Forest

features = ['Age', 'Gender', 'MaritalStatus', 'PartTimeFullTime', 'Dependents']

target = 'InitialIncurredCalimsCost'

x = df[features]
y = df[target]

# Split data  into test and train

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

## Training Random forest Model

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

## Feature Importance

feature_importance = rf.feature_importances_

## Create data frame

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
    })

## sort by importance

importance_df = importance_df.sort_values(by='Importance', ascending=False)

# plotting importance

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance by Random forest')
plt.show()


#
#

## Coefficient - Linear Regression

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)


### Train
 
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Coefficients

coefficients = lr.coef_

# create data frame

importance_coeff_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
    })

# sort by absolute value of coefficinet

importance_coeff_df['Absolute Coefficient'] = importance_coeff_df['Coefficient'].abs()
importance_coeff_df = importance_coeff_df.sort_values(by='Absolute Coefficient', ascending=False)

## Plot Importance 

plt.figure(figsize=(10,6))
sns.barplot(x='Absolute Coefficient', y='Feature', data=importance_coeff_df)
plt.title('Feature Importance (Coefficient-Based)')
plt.show()


#
#
#
#
#


# MODELING 

#
#
#
#

## Data Preparation

model_features = ['DateReported', 'Age', 'Gender', 'MaritalStatus', 'WeeklyWages', 'Dependents']

df['DateReported'] = pd.to_datetime(df['DateReported'])


## date reported to Numerical format, days form ref_date

ref_date = df['DateReported'].min()

df['DaysFromRef'] = (df['DateReported'] - ref_date).dt.days

# replace Datereported

model_features.remove('DateReported')

model_features.append('DaysFromRef')


# select data for modeling

x = df[model_features]

y = df[target]

x_train, y_test, x_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

# Initialize Model

model = LinearRegression()

# Fit model to training data

model.fit(x_train, y_train)

#
#
#

# Evaluate Model

y_pred = model.predict(x_test)

# evaluation

lr_mse = mean_squared_error(y_test, y_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, y_pred)

# model coefficients

model_coeff = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])

# Plot predicted vs actual values

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', lw=2)
plt.xlabel('Actual Claim Cost')
plt.ylabel('Predicted Claim Cost')
plt.title('Actual vs Predicted Claim Cost')
plt.show()












