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

        # Базовые случаи
        if n_samples == 0:
            return None
        if len(unique_classes) == 1:
            return unique_classes[0]
        if self.max_depth is not None and depth >= self.max_depth:
            return self._most_common_class(y)

        # Находим лучшее разбиение
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return self._most_common_class(y)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

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
        parent_entropy = self._entropy(y)
        
        n_left = np.sum(left_indices)
        n_right = np.sum(right_indices)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        child_entropy = (n_left / (n_left + n_right)) * self._entropy(y[left_indices]) + \
                        (n_right / (n_left + n_right)) * self._entropy(y[right_indices])
        
        return parent_entropy - child_entropy

    def _entropy(self, y):
        class_probs = np.bincount(y) / len(y)
        class_probs = class_probs[class_probs > 0]
        
        return -np.sum(class_probs * np.log2(class_probs))

    def _most_common_class(self, y):
        return np.bincount(y).argmax()

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
        return node  # Возвращаем предсказанный класс (значение в листе)
    
    @property
    def feature_importances_(self):
        # Вычисляем важность признаков как среднее значение важностей деревьев
        importances = np.mean([tree.feature_importances_ for tree in self.trees], axis=0)
        
        return importances
