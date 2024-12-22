# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from trees.RandomForestClassifier import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Исследовательский анализ данных (EDA)

# Загрузка данных из файла CSV
data = pd.read_csv('data.csv')

# Просмотр первых 5 строк данных
print(data.head())

# Общая информация о данных
print(data.info())

# Описание статистических характеристик
print(data.describe())

# Визуализация распределения целевой переменной 'Exited'
sns.countplot(x='Exited', data=data)
plt.title('Распределение целевой переменной (Exited)')
plt.show()

# Проверка на пропуски в данных
print(data.isnull().sum())

# 2. Предварительная обработка данных

# Удаление ненужных столбцов
data = data.drop(columns=['id', 'CustomerId', 'Surname'])

# Преобразование категориальных признаков в числовые (Geography и Gender)
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Корреляционная матрица для числовых признаков
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

# Определение признаков и целевой переменной
X = data.drop('Exited', axis=1)  # Все признаки, кроме целевой переменной
y = data['Exited']  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=42)

# 3. Построение и оценка модели

# Если модель уже обучена и сохранена
# Загрузка модели
#model = load('model.joblib')
#from goto import goto, label
#goto .start

# Создание модели Random Forest
model = RandomForestClassifier(n_estimators=10, max_depth=3)

# Обучение модели
model.fit(X_train, y_train)

from joblib import dump, load

# Сохранение модели
dump(model, 'model.joblib')

#label .start

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# Визуализация важности признаков
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Важность признаков')
plt.show()
