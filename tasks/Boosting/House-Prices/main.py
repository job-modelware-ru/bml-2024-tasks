# Импортируем библотеки
import pandas as pd  # Работа с таблицами
import matplotlib.pyplot as plt  # Визуализация
import seaborn as sns  # Визуализация
from sklearn.preprocessing import StandardScaler  # Масштабирование признаков
from sklearn.model_selection import train_test_split  # Разбиение на обучающую и тестовую выборки
import xgboost as xgb  # XGBoost модель
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score  # Метрики


# Загрузка данных
data = pd.read_csv('data.txt', header=0, delimiter=',')


# 1. Исследовательский анализ данных (EDA)

# Первичный осмотр данных
print(data.info())
print(data.describe())

# Построение гистограммы для целевой переменной
plt.figure(figsize=(10, 6))
sns.histplot(data['SalePrice'], kde=True)
plt.title('Распределение цен на жилье')
plt.show()

# Построение корреляционной матрицы числовых признаков
numerical_data = data.select_dtypes(include=['float64', 'int64'])
corr_matrix = numerical_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')
plt.show()


# 2. Предварительная обработка данных

# Отбираем числовые и категориальные столбцы
numerical_data = data.select_dtypes(include=['float64', 'int64'])
categorical_data = data.select_dtypes(include=['object'])

# Заполнение пропущенных значений
data[numerical_data.columns] = numerical_data.fillna(numerical_data.median())
data[categorical_data.columns] = categorical_data.fillna(categorical_data.mode().iloc[0])

# One-Hot Encoding для категориальных признаков
data_encoded = pd.get_dummies(data, drop_first=True)

# Разделение данных на признаки и целевую переменную
X = data_encoded.drop('SalePrice', axis=1)
y = data_encoded['SalePrice']

# Масштабирование признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 3. Построение модели XGBoost

# Инициализация модели
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6)

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Оценка производительности
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f"R²: {r2}")

# Визуализация предсказанных и реальных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.title('Real vs Predicted SalePrice')
plt.show()

