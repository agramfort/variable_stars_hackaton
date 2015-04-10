from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):

    def __init__(self, rf_max_depth=10, rf_n_estimators=50, n_estimators=50, n_jobs=1):
        self.rf_max_depth = rf_max_depth
        self.rf_n_estimators = rf_n_estimators
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.clf = Pipeline([
            ('rf', AdaBoostClassifier(
                base_estimator=RandomForestClassifier(
                    max_depth=self.rf_max_depth, n_estimators=self.rf_n_estimators,
                    n_jobs=self.n_jobs),
                n_estimators=self.n_estimators)
             )
        ])
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
