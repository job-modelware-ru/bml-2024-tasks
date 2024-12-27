import numpy as np
from trees.DecisionTree import DecisionTree

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Случайная выборка с возвращением
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Получаем предсказания от всех деревьев и выбираем наиболее частый класс
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Для каждого образца выбираем наиболее частый класс среди предсказаний деревьев
        return [np.bincount(pred).argmax() for pred in predictions.T]

    @property
    def feature_importances_(self):
        # Вычисляем важность признаков как среднее значение важностей деревьев
        importances = np.mean([tree.feature_importances_ for tree in self.trees], axis=0)
        
        return importances
