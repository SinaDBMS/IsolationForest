# This scripts is an example of how one could use IsolationForest with string features without any data preprocessing.

import pandas as pd

from darian.learn.isolation_forest import IsolationForest
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    # Data preparation
    dataframe: pd.DataFrame = pd.read_csv("../resources/Datasets/Spam_Messages.csv")

    # Labeling: We treat spams as normal instances (0) because of the fact that they exhibit the same structure and
    # Hams as outliers (1).
    dataframe['Category'] = dataframe['Category'].apply(lambda r: 1 if r == 'ham' else 0)

    # Training only on inliers, i.e. Spams
    inliers = dataframe[dataframe['Category'] == 0]

    # Training & scoring samples
    model = IsolationForest(n_estimators=100, max_samples=256,
                            string_features_parameters={"Message": {"splitter": "\s"}})
    # Excluding category from columns, since it contains the label.
    model.fit(inliers, excluded_columns=["Category"])
    scores = model.score_samples(dataframe)

    # Scoring & Evaluation
    print("ROC AUC:", roc_auc_score(dataframe["Category"], scores))  # ROC AUC > 90%
