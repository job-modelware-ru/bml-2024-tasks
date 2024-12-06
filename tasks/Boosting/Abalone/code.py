import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import os

data_path = os.path.join(os.path.dirname(__file__), 'data.txt')

try:
    data = pd.read_csv(data_path, sep=',', engine='python')
except FileNotFoundError:
    print("Ошибка: Файл data.txt не найден. Убедитесь, что файл находится в той же директории, что и скрипт.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Ошибка: Файл data.txt пуст.")
    exit(1)
except pd.errors.ParserError:
    print("Ошибка: Не удалось разобрать файл data.txt. Проверьте формат файла и разделитель.")
    exit(1)

if 'Whole weight.1' in data.columns:
    data = data.drop(columns=['Whole weight.1'])
elif 'Whole weight.2' in data.columns:
    data = data.drop(columns=['Whole weight.2'])

train_test_ratio = 0.8
train_size = int(len(data) * train_test_ratio)
train_set = data[:train_size].copy()
test_set = data[train_size:].copy()

if 'id' not in train_set.columns:
    train_set['id'] = range(len(train_set))
if 'id' not in test_set.columns:
    test_set['id'] = range(len(test_set))

# EDA - Исследовательский анализ данных
sns.countplot(x='Sex', data=train_set)
plt.show()
sns.countplot(x='Rings', data=train_set)
plt.show()

cols = train_set.drop(['id', 'Rings', 'Sex'], axis=1).columns
plt.figure(figsize=(10, 5))
for i, col in enumerate(cols):
    plt.subplot(2, 4, i + 1)
    sns.histplot(x=col, data=train_set)
    plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
for i, col in enumerate(cols):
    sns.boxplot(x='Rings', y=col, data=train_set)
plt.show()

results = {}
for col in cols:
    groups = [group[col] for name, group in train_set.groupby('Rings')]
    test_stat, p_value = kruskal(*groups)
    results[col] = {'Test Statistic': test_stat, 'p-value': p_value}
print(results)

# Предобработка данных
train_set['Sex_coded'], _ = pd.factorize(train_set['Sex'])
test_set['Sex_coded'], _ = pd.factorize(test_set['Sex'])
train_set.drop('Sex', axis=1, inplace=True)
test_set.drop('Sex', axis=1, inplace=True)

X = train_set.drop(['id', 'Rings'], axis=1)
y = train_set['Rings']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_set.drop(['id', 'Rings'], axis=1) # Удаляем Rings из X_test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Построение и оценка модели AdaBoost
model = AdaBoostRegressor(random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_val_scaled)
mse = mean_squared_error(y_val, predictions)
print(f"Mean Squared Error: {mse}")

# Предсказание на тестовом наборе
test_predictions = model.predict(X_test_scaled)

submission = pd.DataFrame({'id': test_set['id'], 'Rings': test_predictions})
submission.to_csv('submission.csv', index=False)
print("submission.csv успешно создан!")
