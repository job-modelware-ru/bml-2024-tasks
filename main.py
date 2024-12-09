import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score

# Импорт и анализ данных
data = pd.read_csv('data.txt')
data.head()

# Обработка пропущенных значений
data.isnull().sum()
data.info()

# Обзор числовых признаков
plt.figure(figsize=(14, 12))
plt.title('Сorrelation matrix')
sns.heatmap(data.corr().round(2), annot=True, cmap='coolwarm')
plt.show()

# Построение модели
# Масштабирование данных
scaler = MinMaxScaler()
X = data.drop(['id', 'smoking'], axis=1)
X_scaled = scaler.fit_transform(X)

# Разделение данных на тренировочную и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X_scaled, data['smoking'], test_size=0.2, random_state=42)

# Используем логистическую регрессию
logreg = LogisticRegression(max_iter=1000)  # Увеличиваем количество итераций для сходимости
logreg.fit(X_train, y_train)

# Предсказания на валидационной выборке
y_pred_logreg = logreg.predict(X_val)

# Оценка работы модели
print('Accuracy:', accuracy_score(y_val, y_pred_logreg))
print('Precision:', precision_score(y_val, y_pred_logreg))
print('Recall:', recall_score(y_val, y_pred_logreg))
print('F1-score:', f1_score(y_val, y_pred_logreg))
print('Confusion Matrix:\n', confusion_matrix(y_val, y_pred_logreg))