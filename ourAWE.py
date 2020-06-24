import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.model_selection import KFold


class OurAWE(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator = None, n_estimators = 10, n_kfsplits = 5):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_kfsplits = n_kfsplits

    def fit(self, X, y):
        self.partial_fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        X, y = check_X_y(X, y)
        
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        if not hasattr(self, 'ensemble_'):
            self.ensemble_ = []

        self.X_ = X
        self.y_ = y

        #### TO BE CONTINUED


    def predict(self, X):
        check_is_fitted(self, "classes_")   #### SPRAWDZENIE CZY JEST NAUCZONY
        X = check_array(X)                  #### SPRAWDZENIE WEJŚCIA
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("Liczba cech się nie zgadza")


        #### TO BE CONTINUED