# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix

# # Загрузка данных
# data = pd.read_csv('data.csv')

# # Проверка наличия пропущенных значений
# print("Пропущенные значения в данных:")
# print(data.isnull().sum())

# # Кодирование целевой переменной
# data['booking_status'] = data['booking_status'].astype('int')  # Убедитесь, что статус отмены - это 0 или 1

# # Определение признаков и целевой переменной
# X = data.drop(['id', 'booking_status'], axis=1)  # Исключим ненужные колонки
# y = data['booking_status']

# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
#     X, y, data.index, test_size=0.2, random_state=42
# )

# # Масштабирование признаков
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Обучение модели SVM
# model = SVC(probability=True)  # Установите probability=True для получения вероятностей
# model.fit(X_train, y_train)

# # Прогнозирование статусов и их вероятностей
# y_pred = model.predict(X_test)
# y_probs = model.predict_proba(X_test)  # Получаем вероятности

# # Создание DataFrame для вывода
# output = pd.DataFrame({
#     'id': data.loc[indices_test, 'id'],  # Получаем идентификаторы
#     'actual_status': y_test,
#     'predicted_status': y_pred,
#     'probability_cancel': y_probs[:, 1],  # Вероятность отмены
# })

# # Сохранение результатов в CSV файл
# output.to_csv('output.csv', index=False)

# # Вывод матрицы ошибок и отчета о классификации
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np 

# Загрузка данных
df = pd.read_csv('data.csv')
df.head()

# Определим размеры датасета
print("Размеры датасета:", df.shape)

# Сделаем столбец 'id' индексом данных
df = df.set_index('id', drop=True)

# Вывод первых строк DataFrame для проверки
print(df.head())

# Проверка пропущенных значений
all_data_na = (df.isnull().sum() / len(df)) * 100

# Удаление признаков без пропущенных значений и сортировка по убыванию
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index)

# Проверка, остались ли признаки после фильтрации
if not all_data_na.empty:
    all_data_na = all_data_na.sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    print(missing_data)

    # Построение графика пропусков
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='vertical')
    sns.barplot(x=all_data_na.index, y=all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()
else:
    print("Нет признаков с пропущенными значениями.")

# Разделяем данные на обучающую и тестовую выборки
# test_size=0.2 означает, что 20% данных будет отведено для тестирования
# random_state=42 устанавливает фиксированный начальный случай для воспроизводимости результатов
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Выводим первые несколько строк обучающей выборки для предварительного просмотра
print(train_df.head())

# Обзор целевой переменной
print(train_df['booking_status'].describe())

# визуализация
f, ax = plt.subplots(figsize=(10, 8))
sns.histplot(train_df['booking_status'])
plt.show()

# Рассчитываем асимметрию и эксцесс
print("Ассиметрия: %f" % train_df['booking_status'].skew())
print("Эксцесс: %f" % train_df['booking_status'].kurt())
# Необходимо выполнить преобразование целевой переменной

numeric_df = train_df.select_dtypes(include=[np.number])

# Вычисляем матрицу корреляции на числовых данных
corr_matrix = numeric_df.corr()

# Устанавливаем размер фигуры для тепловой карты
plt.figure(figsize=(20, 18))

# Построение тепловой карты корреляции с аннотациями
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',annot_kws={'size': 10}, square=True, linewidths=.5)

# Добавляем заголовок
plt.title('Матрица корреляции числовых признаков', fontsize=16)

# Отображаем график
plt.show()
k = 9 # количество коррелирующих признаков, которое мы хотим увидеть
cols = corr_matrix.nlargest(k, 'booking_status')['booking_status'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 
                 fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

def find_strong_correlations(train_df, feature_list, target='booking_status', threshold=0.5):
    # Списки для хранения результатов
    features_with_high_corr = []
    features_to_add = []
    pairs = set()
    # Перебираем все пары признаков из двух списков
    for f1 in feature_list:
        for f2 in feature_list:
            # Чтобы не сравнивать признак с самим собой
            if f1 != f2 and target not in [f1, f2] and (f1, f2) not in pairs and (f2, f1) not in pairs:
                # Если корреляция между признаками больше порога
                if abs(corr_matrix.loc[f1, f2]) > threshold:
                    # Находим корреляцию с целевым признаком 'booking_status'
                    corr_f1_target = abs(corr_matrix.loc[f1, target])
                    corr_f2_target = abs(corr_matrix.loc[f2, target])

                    # Добавляем в список тот признак, который сильнее коррелирует с 'booking_status'
                    if corr_f1_target > corr_f2_target:
                        features_with_high_corr.append(f1)
                        features_to_add.append(f2)
                    else:
                        features_with_high_corr.append(f2)
                        features_to_add.append(f1)
                pairs.add((f1, f2))

    # Возвращаем два списка: признаки с высокой корреляцией с 'booking_status' и оставшиеся признаки
    return features_with_high_corr, features_to_add

feature_num = list(train_df.select_dtypes(include=[np.number]).columns)
high_corr_features, other_features = find_strong_correlations(train_df, feature_num)

print("Признаки с сильной корреляцией с booking_status:", high_corr_features)
print("Оставшиеся признаки из пары (с более слабой корреляцией):", other_features)
test_df = test_df.drop(columns=other_features)
train_df = train_df.drop(columns=other_features)

feature_num = list(train_df.select_dtypes(include=[np.number]).columns)

# Рассчитываем корреляцию между всеми признаками
selected_df = train_df[feature_num]
corr_matrix = selected_df.corr()
plt.figure(figsize=(20, 18))

# Построение тепловой карты корреляции с аннотациями
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',annot_kws={'size': 10}, square=True, linewidths=.5)

# Добавляем заголовок
plt.title('Матрица корреляции числовых признаков', fontsize=16)

# Отображаем график
plt.show()

numeric_df = train_df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
# Дополнительно: Вывод наиболее коррелированных признаков с целевой переменной
plt.figure(figsize=(12, 10))
# Сортируем по корреляции с booking_status
top_corr = corr_matrix['booking_status'].sort_values(ascending=False)
sns.barplot(x=top_corr.values, y=top_corr.index)
plt.title('Корреляция числовых признаков с booking_status', fontsize=16)
plt.xlabel('Коэффициент корреляции')
plt.ylabel('Признаки')
plt.show()