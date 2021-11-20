import pandas as pd

from darian.learn.isolation_forest import IsolationForest
from sklearn.ensemble import IsolationForest as SKLearnIsolationForest
from sklearn.datasets import fetch_kddcup99
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    # Data preparation
    categorical_features = ['protocol_type', 'service', 'flag']
    X: pd.DataFrame
    y: pd.Series
    X, y = fetch_kddcup99(as_frame=True, return_X_y=True)

    # Interpreting categorical features
    for categorical_feature in categorical_features:
        X[categorical_feature] = X[categorical_feature].astype('category')

    # Interpreting numerical features
    for numerical_feature in X.columns.difference(categorical_features):
        X[numerical_feature] = X[numerical_feature].astype('float')

    # Get a one hot encoding the categorical features in order to be used by the algorithms which only accept
    # numerical features
    one_hot_encoded_X = pd.get_dummies(X)

    # Labeling: We'd like to interpret all instances that have some label other than normal as outlier.
    y = y.apply(lambda r: 0 if r == b'normal.' else 1)

    # We'd like to train our model over normal data points. For that reason we exclude the outliers from training data.
    training_data = X[y == 0]
    one_hot_encoded_training_data = one_hot_encoded_X[y == 0]

    # Training and scoring samples
    model_1 = IsolationForest(n_estimators=100, max_samples=256, n_jobs=11)
    model_1.fit(training_data)
    scores_1 = model_1.score_samples(X)

    model_2 = SKLearnIsolationForest(n_estimators=100, max_samples=256, n_jobs=11)
    model_2.fit(one_hot_encoded_training_data)
    scores_2 = model_2.score_samples(one_hot_encoded_X) * -1

    # Evaluation
    print("ROC AUC of IsolationForest with support for categorical features:", roc_auc_score(y, scores_1))
    print("ROC AUC of SKLearn's IsolationForest", roc_auc_score(y, scores_2))
