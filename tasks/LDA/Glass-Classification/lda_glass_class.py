import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Исследовательский анализ данных (EDA)
# Загрузка данных
data = pd.read_csv("data.csv")

# Просмотр первых строк и общей информации
def explore_data(data):
    print("Первые 5 строк данных:\n", data.head())
    print("\nИнформация о данных:")
    print(data.info())
    print("\nСтатистика данных:\n", data.describe())
    print("\nПроверка на пропущенные значения:\n", data.isnull().sum())

explore_data(data)

# Визуализация
plt.figure(figsize=(12, 6))
sns.countplot(x='Type', data=data, color='#B0C4DE')
plt.title('Распределение типов стекла')
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Корреляция между признаками')
plt.show()

# 2. Предварительная обработка данных
# Разделение данных на признаки и целевую переменную
X = data.drop('Type', axis=1)
y = data['Type']

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. Построение и оценка модели
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Предсказание
y_pred = lda.predict(X_test)

# Оценка модели
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Визуализация
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
