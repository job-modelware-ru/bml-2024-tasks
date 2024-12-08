from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import init, Fore, Back

init(autoreset=True)

from read_data import data

# 1. Исследовательский анализ данных (EDA)
# Просмотр первых строк данных
print(Fore.LIGHTCYAN_EX + "1. Исследовательский анализ данных (EDA)\n")
print("__________________Просмотр первых строк данных__________________\n", data.head(), "\n\n")

# Проверка на пропуски
print("__________________Проверка на пропуски__________________\n", data.isnull().sum(), "\n\n")

# Визуализация распределения целевой переменной
sns.countplot(x='NObeyesdad', data=data, color="lightblue")
plt.title('Distribution of Obesity Levels')
plt.show()

# Вывод уникальных значений по ключам
print("__________________Вывод уникальных значений по ключам__________________\n")

print("len(data.columns): ", len(data.columns), "\n\n")

keys = data.columns
for column in keys:
        unique_vals = data[column].unique()
        print(f"unique_vals for [{column}]: ", unique_vals, "\n")


# 2. Предварительная обработка данных
# Преобразование категориальных переменных в числовые
print(Fore.LIGHTCYAN_EX + "\n\n2. Предварительная обработка данных\n")
print("__________________Преобразование категориальных переменных в числовые__________________\n")
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['family_history_with_overweight'] = data['family_history_with_overweight'].map({'yes': 1, 'no': 0})
data['FAVC'] = data['FAVC'].map({'yes': 1, 'no': 0})
data['NObeyesdad'] = data['NObeyesdad'].map({
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
})
data['CALC'] = data['CALC'].map({'Sometimes': 1, 'no': 0, 'Frequently': 2})
data['CAEC'] = data['CALC'].map({'Sometimes': 1, 'no': 0, 'Frequently': 2, 'Always': 3})
data['SMOKE'] = data['SMOKE'].map({'yes': 1, 'no': 0})
data['SCC'] = data['SMOKE'].map({'yes': 1, 'no': 0})
data['MTRANS'] = data['MTRANS'].map({'Public_Transportation': 0, 'Automobile': 1, 'Walking': 2, 'Motorbike': 3, 'Bike': 4})

keys = data.columns
for column in keys:
        unique_vals = data[column].unique()
        print(f"unique_vals for [{column}]: ", unique_vals, "\n")


# Разделение данных на признаки и целевую переменную
X = data.drop(['id', 'NObeyesdad'], axis=1)
y = data['NObeyesdad']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Построение и оценка модели
print(Fore.LIGHTCYAN_EX + "\n\n3. Построение и оценка модели\n")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

plt.hist(y_pred, bins=7, rwidth=0.8, color="lightblue")
plt.show()


# Оценка модели
print("__________________Оценка модели__________________\n")
print("confusion_matrix\n", confusion_matrix(y_test, y_pred))
print("\n\nclassification_report\n", classification_report(y_test, y_pred))



