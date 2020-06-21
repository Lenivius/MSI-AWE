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
        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if not hasattr(self, 'ensemble_'):
            self.ensemble_ = []

        self.X_ = X
        self.y_ = y



        return self


    def predict(self, X):
        #### TO BE DONE


    def ensemble_support_matrix(self, X):
        #### TO BE DONE