"""
Model evaluation utilities: metric computation and comparison table generation.
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(name, pipeline_label, model, X_test, y_test):
    """Compute Precision, Recall, F1, and AUC-ROC for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'Model': name,
        'Pipeline': pipeline_label,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }


def compute_leakage_drop(results_df, metric='F1'):
    """Compute the % drop in a metric between the Leaky and Clean pipelines, per model."""
    pivot = results_df.pivot(index='Model', columns='Pipeline', values=metric)
    pivot[f'{metric}_drop_%'] = ((pivot['Leaky'] - pivot['Clean']) / pivot['Leaky']) * 100
    return pivot
