import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.model_selection import StratifiedKFold


class OurAWE(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator = None, n_estimators = 10, n_skfsplits = 5, rnd_state = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_skfsplits = n_skfsplits
        self.rnd_state = rnd_state
        self.shuffle = False

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
        if not hasattr(self, 'weights_'):
            self.weights_ = []

        self.X_ = X
        self.y_ = y

        new_estimator = clone(self.base_estimator).fit(self.X_, self.y_)

        p_c = np.unique(self.y_, return_counts = True)[1] / len(self.y_)
        MSEr = np.sum(p_c * (1 - p_c) ** 2)

        if self.rnd_state is not None:
            self.shuffle = True
        skf = StratifiedKFold(n_splits = self.n_skfsplits, random_state = self.rnd_state, shuffle = self.shuffle)

        scores = []
        for train, test in skf.split(X, y):
            fold_estimator = clone(self.base_estimator).fit(self.X_[train], self.y_[train])
            scores.append(self.calc_MSEi(fold_estimator, self.X_[test], self.y_[test]))


        new_estimator_MSEi = np.mean(scores)
        new_estimator_weight = MSEr - new_estimator_MSEi

        
        for clf_id, clf in enumerate(self.ensemble_):
            clf_weight = MSEr - self.calc_MSEi(clf, self.X_, self.y_)
            self.weights_[clf_id] = clf_weight

        
        self.ensemble_.append(new_estimator)
        self.weights_.append(new_estimator_weight)

        if len(self.ensemble_) > self.n_estimators:
            min = self.weights_[0]
            for w in self.weights_:
                if w < min:
                    min = w
            index = self.weights_.index(min)
            del self.ensemble_[index]
            del self.weights_[index]


    def calc_MSEi(self, clf, X_, y_):
        pred_prob = clf.predict_proba(X_)
        prob = np.zeros(len(y_))
        for y_id in range (len(y_)):
            for label in self.classes_:
                if y_[y_id] == label:
                    prob[y_id] = pred_prob[y_id, label]

        return np.sum((1 - prob) ** 2)


    def ensemble_support_matrix(self, X):
        probas_ = []
        for member_clf in self.ensemble_:
            probas_.append(member_clf.predict_proba(X))

        return np.array(probas_)


    def predict(self, X):
        check_is_fitted(self, "classes_")   #### SPRAWDZENIE CZY JEST NAUCZONY
        X = check_array(X)                  #### SPRAWDZENIE WEJŚCIA
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("Liczba cech się nie zgadza")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        return self.classes_[prediction]