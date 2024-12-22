import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Базовый случай 1: Если нет примеров
        if n_samples == 0:
            return None

        # Базовый случай 2: Если достигли максимальной глубины
        if self.max_depth is not None and depth >= self.max_depth:
            return self._most_common_class(y)

        # Базовый случай 3: Если все метки одинаковые
        if len(unique_classes) == 1:
            return unique_classes[0]

        # Находим лучшее разбиение
        best_feature, best_threshold = self._best_split(X, y)

        # Если не удалось найти разбиение (например, все значения одинаковы)
        if best_feature is None:
            return self._most_common_class(y)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        # Проверка на наличие пустых подвыборок
        if not np.any(left_indices) or not np.any(right_indices):
            return self._most_common_class(y)

        # Рекурсивно строим поддеревья
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _most_common_class(self, y):
        """Возвращает наиболее распространенный класс в целевой переменной."""
        return np.bincount(y).argmax()

    def _best_split(self, X, y):
        n_features = X.shape[1]
        
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = X[:, feature_idx] < threshold
                right_indices = X[:, feature_idx] >= threshold
                
                if len(np.unique(y[left_indices])) == 0 or len(np.unique(y[right_indices])) == 0:
                    continue
                
                gain = self._information_gain(y, left_indices, right_indices)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _information_gain(self, y, left_indices, right_indices):
        """Вычисляет прирост информации."""
        
        # Полная энтропия перед разбиением
        parent_entropy = self._entropy(y)
        
        # Энтропия после разбиения
        n_left = np.sum(left_indices)
        n_right = np.sum(right_indices)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        child_entropy = (n_left / (n_left + n_right)) * self._entropy(y[left_indices]) + \
                        (n_right / (n_left + n_right)) * self._entropy(y[right_indices])
        
        # Прирост информации
        return parent_entropy - child_entropy

    def _entropy(self, y):
        """Вычисляет энтропию целевой переменной."""
        
        class_probs = np.bincount(y) / len(y)
        
        # Удаляем нулевые вероятности для вычисления логарифма
        class_probs = class_probs[class_probs > 0]
        
        return -np.sum(class_probs * np.log2(class_probs))

    def predict(self, X):
        return np.array([self._predict(sample) for sample in X])

    def _predict(self, sample):
        node = self.tree
        while isinstance(node, tuple):
            feature_idx, threshold, left_subtree, right_subtree = node
            if sample[feature_idx] < threshold:
                node = left_subtree
            else:
                node = right_subtree
        return node

