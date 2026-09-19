"""
Preprocessing utilities: missing value handling, encoding, and time-based splitting.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_target(df, status_col='tr_status'):
    """Map tr_status ('passed'/'failed') to a binary target column."""
    df = df[df[status_col].isin(['passed', 'failed'])].copy()
    df['target'] = (df[status_col] == 'failed').astype(int)
    return df


def fill_missing_values(df):
    """Fill missing values: 0 for PR-related counts, median for numeric, 'unknown' for categorical."""
    for col in ['gh_pull_req_num', 'gh_num_issue_comments', 'gh_num_pr_comments']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].fillna('unknown')

    return df


def time_based_split(df, time_col='gh_build_started_at', train_frac=0.8):
    """Sort by build start time and split chronologically to avoid future-information leakage."""
    df = df.sort_values(time_col)
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def encode_and_scale(X_train, X_test):
    """Label-encode categorical columns and standard-scale numeric columns, fit on train only."""
    X_train, X_test = X_train.copy(), X_test.copy()

    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
        X_test[col] = le.transform(X_test[col])

    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test
