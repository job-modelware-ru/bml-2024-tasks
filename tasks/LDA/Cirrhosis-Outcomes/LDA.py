# Импорт необходимых библиотек: pandas для работы с данными, sklearn для машинного обучения,
# imblearn для обработки несбалансированных данных, seaborn и matplotlib для визуализации.
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Попытка чтения данных из файла 'data.txt'
try:
  data = pd.read_csv('data.txt')
except FileNotFoundError:
  print("Файл 'data.txt' не найден")
  exit()

# Заменяем 'S' на 'Y' в столбце 'Edema' (предполагается, что 'S' означает наличие отёка)
data['Edema'] = data['Edema'].replace({'S': 'Y'})
data['Age_Years'] = data['Age'] / 365.25
# Список категориальных признаков
categorical_cols = ['Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']

# One-Hot Encoding для категориальных признаков
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

le = LabelEncoder()
# Преобразование целевой переменной 'Status' в числовое представление с помощью Label Encoding
data['Status'] = le.fit_transform(data['Status'])

# Целевая переменная
y = data['Status']

# Список числовых столбцов
numeric_cols = data.select_dtypes(include=['number']).columns
for col in numeric_cols:
    # Заполнение пропущенных значений медианой для каждого числового столбца
    data[col] = data[col].fillna(data[col].median())

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(['id', 'Status', 'Drug'], axis=1)

# Разбиение данных на тренировочную и тестовую выборки (90/10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Инициализация SMOTE для обработки несбалансированных данных
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

pipeline = Pipeline([
  ('scale', StandardScaler()), # Стандартизация данных
  ('lda', LinearDiscriminantAnalysis()) # Линейный дискриминантный анализ
])

#  Поиск лучшего решателя для LDA
param_grid = {
  'lda__solver': [ 'lsqr', 'eigen'],
}

# Поиск лучших гиперпараметров с помощью GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=1)

# Обучение модели
grid_search.fit(X_train_resampled, y_train_resampled)

# Лучшая модель после GridSearchCV
best_model = grid_search.best_estimator_
# Предсказания на тестовой выборке
y_pred = best_model.predict(X_test)

# Оценка
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy:.2f}")
print(classification_report(y_test, y_pred)) # Отчет классификации с точностью, полнотой, F1-мерой
print(f"Лучшие параметры: {grid_search.best_params_}") # Лучшие найденные гиперпараметры

# Сохранение результатов в CSV-файл
y_pred_original = le.inverse_transform(y_pred)
y_test_original = le.inverse_transform(y_test)
test_indices = X_test.index
results = pd.DataFrame({'id': data.loc[test_indices, 'id'],
                     'Actual Status': y_test_original,
                     'Predicted Status': y_pred_original})

results.to_csv('prediction_results.csv', index=False)
print("Результаты предсказаний сохранены в prediction_results.csv")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
colors = [ (1, 0.8, 0.8), (0.8, 0.4, 0.8), (0.6, 0, 0.6)] # Светло-розовый, средне-розовый, темно-розовый
cmap = mcolors.LinearSegmentedColormap.from_list("my_pink_cmap", colors)
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
