# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 05:49:57 2024

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

df = pd.read_csv('Final_Health_insurance_data.csv')

# MODELING 

#
#
#

## Data Preparation

model_features = ['DateReported', 'Age', 'Gender', 'MaritalStatus', 'WeeklyWages', 'Dependents']

target = 'InitialIncurredCalimsCost'

df['DateReported'] = pd.to_datetime(df['DateReported'])

df.set_index('ClaimNumber', inplace=True)

## date reported to Numerical format, days form ref_date

ref_date = df['DateReported'].min()

df['DaysFromRef'] = (df['DateReported'] - ref_date).dt.days

# replace Datereported

model_features.remove('DateReported')

model_features.append('DaysFromRef')


# select data for modeling

x = df[model_features]

y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

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








# End of Modeling
