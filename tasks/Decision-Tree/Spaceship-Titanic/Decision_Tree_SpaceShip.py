import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Загрузка данных
spaceship_df = pd.read_csv('data.csv')
print(spaceship_df)
spaceship_df.info()
print(spaceship_df.describe())

# Количество пропусков в каждом столбце
print(spaceship_df.isnull().sum())

#Заполнение пропусков
spaceship_df['Age'] = spaceship_df.groupby(['HomePlanet', 'VIP'])['Age'].transform(lambda x: x.fillna(x.median()))
spaceship_df['CryoSleep'] = spaceship_df.groupby(['HomePlanet', 'VIP'])['CryoSleep'].transform(lambda x: x.fillna(x.mode()[0]))
spaceship_df['Destination'] = spaceship_df.groupby(['HomePlanet', 'VIP'])['Destination'].transform(lambda x: x.fillna(x.mode()[0]))
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    spaceship_df[col] = spaceship_df[col].fillna(spaceship_df[col].median())

# Удаление ненужных столбцов
spaceship_df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)

# Проверка результата
spaceship_df.head()

# EDA
# Создание графиков
figure, axes = plt.subplots(1, 2, figsize=(15, 5))
axis = sns.countplot(spaceship_df, x='HomePlanet', ax=axes[0])
axis.bar_label(axis.containers[0])

sns.barplot(
    spaceship_df,
    x='HomePlanet',
    y='Transported',
    estimator=lambda x: sum(x) * 100.0 / len(x),
    ax=axes[1]
)

axes[0].set_title("Количество пассажиров с каждой планеты")
axes[1].set_title("Процент транспортированных пассажиров")
axes[1].set_ylabel("Процент Transported (%)")
plt.tight_layout()
plt.show()


figure, axes = plt.subplots(1, 2, figsize=(15, 5))
axis = sns.histplot(spaceship_df, x='RoomService', bins=10, kde=False, ax=axes[0])
axis.set_xticks(range(0, int(spaceship_df['RoomService'].max()) + 1, 500))
sns.barplot(
    spaceship_df,
    x=pd.cut(spaceship_df['RoomService'], bins=5, labels=['Очень низкие', 'Низкие', 'Средние', 'Высокие', 'Очень высокие']),
    y='Transported',
    estimator=lambda x: sum(x) * 100.0 / len(x),
    ax=axes[1],
)
axes[0].set_title("Распределение расходов на RoomService")
axes[0].set_ylabel("Количество пассажиров")
axes[1].set_title("Процент транспортированных пассажиров по расходам на RoomService")
axes[1].set_ylabel("Процент Transported (%)")
axes[1].set_xlabel("Уровень расходов на RoomService")
plt.tight_layout()
plt.show()

figure, axes = plt.subplots(1, 2, figsize=(15, 5))
axis = sns.countplot(spaceship_df, x='Destination', ax=axes[0])
axis.bar_label(axis.containers[0])
axis.set_title("Распределение пассажиров по назначению")
axis.set_xlabel("Destination")
axis.set_ylabel("Количество пассажиров")

sns.barplot(
    spaceship_df,
    x='Destination',
    y='Transported',
    estimator=lambda x: sum(x) * 100.0 / len(x),
    ax=axes[1]
)

axes[1].set_title("Процент транспортированных пассажиров по назначению")
axes[1].set_ylabel("Процент Transported (%)")
axes[1].set_xlabel("Destination")
plt.tight_layout()
plt.show()


figure, axes = plt.subplots(1, 2, figsize=(15, 5))
axis = sns.countplot(spaceship_df, x='CryoSleep', ax=axes[0])
axis.bar_label(axis.containers[0])
axis.set_title("Количество пассажиров по статусу CryoSleep")
axis.set_ylabel("Количество пассажиров")
axis.set_xlabel("CryoSleep")

sns.barplot(
    spaceship_df,
    x='CryoSleep',
    y='Transported',
    estimator=lambda x: sum(x) * 100.0 / len(x),
    ax=axes[1]
)

axes[1].set_title("Процент транспортированных пассажиров по статусу CryoSleep")
axes[1].set_ylabel("Процент Transported (%)")
axes[1].set_xlabel("CryoSleep")
plt.tight_layout()
plt.show()


figure, axis = plt.subplots(1, 1, figsize=(15, 5))
sns.histplot(spaceship_df, x='Age', bins=50, kde=True, ax=axis)


axis.set_title("Распределение возраста пассажиров")
axis.set_xlabel("Возраст")
axis.set_ylabel("Количество пассажиров")
plt.tight_layout()
plt.show()


# Преобразование категориальных переменных с использованием pd.get_dummies
spaceship_df = pd.get_dummies(spaceship_df, columns=['CryoSleep', 'Destination', 'HomePlanet'], drop_first=True)
print(spaceship_df)


x = spaceship_df.drop('Transported', axis=1)
y = spaceship_df['Transported']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['Not Transported', 'Transported'],
    yticklabels=['Not Transported', 'Transported'],
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Матрица ошибок')
plt.show()

feature_importances = pd.Series(model.feature_importances_, index=x.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Важность признаков')
plt.show()
