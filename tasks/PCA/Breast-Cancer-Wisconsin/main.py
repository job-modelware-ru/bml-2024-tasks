# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Загрузка данных
data = pd.read_csv("data.csv")

# Удаление ненужных столбцов (если они присутствуют в данных)
if "Unnamed: 32" in data.columns:
    data.drop(columns=["id", "Unnamed: 32"], inplace=True)  # Удаляем столбцы с id и лишними данными
else:
    data.drop(columns=["id"], inplace=True)  # Если "Unnamed: 32" нет, удаляем только id

# Проверка на наличие пропущенных значений
print("\nПропущенные значения:")
print(data.isnull().sum())

# Замена пропущенных значений в числовых столбцах на средние значения
numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# Преобразование целевой переменной в числовую (M -> 1, B -> 0)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(columns=["diagnosis"])  # Признаки
y = data["diagnosis"]  # Целевая переменная

# Нормализация признаков с использованием StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Применение анализа главных компонент (PCA) для уменьшения размерности
pca = PCA(n_components=2)  # Преобразуем данные в двумерное пространство
X_pca = pca.fit_transform(X_scaled)

# Визуализация результатов PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="coolwarm", alpha=0.7)
plt.title("PCA: Визуализация признаков")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.show()

# Логистическая регрессия для классификации
# Разделение данных на обучающую и тестовую выборки (70% обучающие, 30% тестовые)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Обучение модели логистической регрессии
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Предсказание результатов на тестовой выборке
y_pred = logreg.predict(X_test)

# Оценка модели
print("\nОценка логистической регрессии:")
print("Accuracy:", accuracy_score(y_test, y_pred))  # Точность модели
print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred))  # Матрица ошибок
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred))  # Детальный отчет по меткам

# Графическое отображение матрицы ошибок
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", cbar=False)
plt.title("Матрица ошибок")
plt.xlabel("Предсказанные значения")
plt.ylabel("Истинные значения")
plt.show()
