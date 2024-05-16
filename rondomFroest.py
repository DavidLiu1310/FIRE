import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import FIRE

# 读取数据
red_wine = pd.read_csv('./data/wine/winequality-red.csv', sep=';')

# 分割特征和标签
X = red_wine.drop('quality', axis=1)
y = red_wine['quality']

features = X.columns

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=500, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
pearson = pearsonr(y_test, y_pred)[0]

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")
print(f"pearson: {pearson}")

'''
Mean Squared Error: 0.3013528125
Mean Absolute Error: 0.42196875
R² Score: 0.5388674666387723
pearson: 0.7368944640185024

'''