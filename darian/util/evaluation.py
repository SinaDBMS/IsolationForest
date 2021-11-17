from sklearn.metrics import precision_recall_curve, auc


def precision_recall_auc_score(true_labels, predictions) -> float:
    precision, recall, threshold = precision_recall_curve(true_labels, predictions)
    return auc(recall, precision)
