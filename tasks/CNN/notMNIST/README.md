### notMNIST

__Цель__ - классифицировать изображения букв в 10 классов

__Модель:__ CNN


__Загрузка данных:__ 

```
import torch
import matplotlib.pyplot as plt

loaded_data = torch.load('notMNIST.pt', weights_only=True)

dataset_tensor = loaded_data['images']
labels_tensor = loaded_data['labels']

plt.imshow(dataset_tensor[6, :, :])
plt.show()
print(labels_tensor[6])
```
