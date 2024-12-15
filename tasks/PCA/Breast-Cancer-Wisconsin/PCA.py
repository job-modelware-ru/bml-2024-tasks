# Импортирование библиотек
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Чтение данных
data = pd.read_csv("data.csv")

# Предобработка данных
data.drop(columns=["id"], inplace=True)  # Удаление столбца ID
data.drop(columns=["Unnamed: 32"], inplace=True)  # Удаление столбца Unnamed: 32
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})  # Замена переменных на числа

print(data.isnull().sum())  # Проверка пропущенных значений

# Обработка данных
X = data.drop(columns=["diagnosis"])
Y = data["diagnosis"]
scl = StandardScaler()
X_scl = scl.fit_transform(X)  # Нормализация

# Применение PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scl)

# Визуализация результатов
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=Y, palette="cool", alpha=0.7)
plt.title("PCA")
plt.xlabel("M (1)")
plt.ylabel("B (0)")
plt.show()

# Применение Logistic Regression
X_train, X_test, Y_train, Y_test = train_test_split(X_scl, Y, test_size=0.4, random_state=38)  # Тренировочная и тестовая выборки
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)  # Обучение модели
Y_pred = logreg.predict(X_test)  # Предсказанное значение

# Вывод оценки модели
print("\nТочность:")
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nМатрица ошибок:")
print(confusion_matrix(Y_test, Y_pred))
print("\nОтчет классификации:")
print(classification_report(Y_test, Y_pred))
