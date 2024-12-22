import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Загрузка данных
all_data = pd.read_csv('data.txt')

# Анализ структуры данных
print(all_data.info(show_counts=True))
print(all_data.head())

# Предварительная обработка данных
all_data = all_data.dropna().drop('id', axis=1)
print(all_data.info())

target_column = 'loss'

# Анализ распределения целевой переменной
plt.figure(figsize=(12, 6))
sns.histplot(all_data[target_column], kde=True)
plt.title('Распределение потерь')
plt.show()

# Удаление выбросов
Q1 = all_data.quantile(0.25)
Q3 = all_data.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

all_data = all_data.mask((all_data < lower_bound) | (all_data > upper_bound), all_data.median(), axis=1)

# Корреляционный анализ
plt.figure(figsize=(12, 8))
sns.heatmap(all_data.corr(), annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

# Отбор признаков на основе корреляции
correlation = all_data.corr()[target_column]
n = 15 + 1  # количество наиболее коррелирующих переменных + 1
cor_vals = correlation.abs().sort_values(ascending=False)
top_n_correlations = cor_vals.head(n)
correlated_keys = top_n_correlations.keys().tolist()
corr_data = all_data[correlated_keys]
print(corr_data.info())

# Подготовка данных для обучения
prep_data = corr_data
X = prep_data.drop(target_column, axis=1)
y = prep_data[target_column]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение модели XGBoost
xgboost_model = XGBRegressor(random_state=42, learning_rate=0.1, max_depth=6, n_estimators=100)
xgboost_model.fit(X_train, y_train)

y_pred = xgboost_model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
