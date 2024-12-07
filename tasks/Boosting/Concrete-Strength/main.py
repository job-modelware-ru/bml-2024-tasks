# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Исследовательский анализ данных (EDA)

# Загрузка данных
data = pd.read_csv('data.csv')

# Проверка имен столбцов
print("Имена столбцов:")
print(data.columns)

# Первые строки данных
print("\nПервые строки данных:")
print(data.head())

# Описание данных
print("\nОписание данных:")
print(data.describe())

# Проверка наличия пропущенных значений
print("\nПропущенные значения:")
print(data.isnull().sum())

# Распределение целевой переменной
# Проверка имени столбца перед использованием
target_column = 'Strength'
if target_column in data.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[target_column], kde=True)
    plt.title('Распределение прочности бетона')
    plt.show()
else:
    print(f"Столбец '{target_column}' не найден в данных.")

# Корреляционная матрица
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

# 2. Предварительная обработка данных

# Отбираем числовые и категориальные столбцы
numerical_data = data.select_dtypes(include=['float64', 'int64'])
categorical_data = data.select_dtypes(include=['object'])

# Заполнение пропущенных значений
data[numerical_data.columns] = numerical_data.fillna(numerical_data.median())

# Проверка на наличие категориальных столбцов
if not categorical_data.empty:
    data[categorical_data.columns] = categorical_data.fillna(categorical_data.mode().iloc[0])

# Сохраняем исходные имена признаков
original_feature_names = data.columns.tolist()

# One-Hot Encoding для категориальных признаков
data_encoded = pd.get_dummies(data, drop_first=True)

# Разделение данных на признаки и целевую переменную
if target_column in data_encoded.columns:
    X = data_encoded.drop(target_column, axis=1)
    y = data_encoded[target_column]
else:
    print(f"Столбец '{target_column}' не найден в данных. Прерывание выполнения.")
    exit()

# Масштабирование признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Построение и оценка модели

# Создание объекта модели
model = xgb.XGBRegressor()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка производительности
print("\nОценка производительности:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Настройка гиперпараметров
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 300, 1000]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Лучшие параметры
print("\nЛучшие параметры:")
print("Best Parameters:", grid_search.best_params_)

# Оценка лучшей модели
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\nОценка лучшей модели:")
print("Best Model Mean Squared Error:", mean_squared_error(y_test, y_pred_best))
print("Best Model R^2 Score:", r2_score(y_test, y_pred_best))

# Лучшее предсказанное значение
print("\nЛучшее предсказанное значение:")
print("Best Predicted Value:", y_pred_best.mean())

