import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 1. Загрузка данных
# ------------------
loaded_data = torch.load('notMNIST.pt', weights_only=True)

# Получаем тензоры изображений и меток
dataset_tensor = loaded_data['images']  # тензор (N, 28, 28)
labels_tensor = loaded_data['labels']   # тензор (N,)

# Визуализируем пример изображения
plt.imshow(dataset_tensor[6, :, :], cmap='gray')
plt.title(f"Метка: {labels_tensor[6]}")
plt.show()

# Преобразуем тензор изображений в формат (N, 784)
images = dataset_tensor.view(dataset_tensor.shape[0], -1).numpy()
labels = labels_tensor.numpy()

print(f"Размерность данных: {images.shape}, Метки: {labels.shape}")

# 2. Понижение размерности с t-SNE
# --------------------------------
# Используем часть данных для ускорения работы t-SNE
# Количество данных
total_size = images.shape[0]

# Выбираем максимум 5000 примеров или 20% данных
sample_size = min(int(0.2 * total_size), 5000)
# Берём случайные индексы для выборки
indices = np.random.choice(total_size, sample_size, replace=False)
images_sample = images[indices]
labels_sample = labels[indices]

print("Понижение размерности с помощью t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
images_tsne = tsne.fit_transform(images_sample)

# Визуализация кластеров t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(images_tsne[:, 0], images_tsne[:, 1], c=labels_sample, cmap='tab10', s=10)
plt.legend(*scatter.legend_elements(), title="Классы")
plt.title("Визуализация t-SNE на notMNIST")
plt.xlabel("Первая компонента")
plt.ylabel("Вторая компонента")
plt.show()

# 3. Обучение KNN
# ----------------
# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Обучаем KNN-классификатор
print("Обучение KNN-классификатора...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Делаем предсказания на тестовых данных
y_pred = knn.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность KNN: {accuracy:.4f}")

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title("Матрица ошибок KNN")
plt.xlabel("Предсказанные метки")
plt.ylabel("Истинные метки")
plt.show()
