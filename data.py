import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import (
    DEFOG_DIR, TDCSFOG_DIR,
    FEATURES, LABELS,
    WINDOW_SIZE, STEP_SIZE,
    BATCH_SIZE, TEST_SIZE, RANDOM_SEED,
)


def load_defog() -> pd.DataFrame:
    dfs = []
    for f in glob.glob(f'{DEFOG_DIR}*.csv'):
        df = pd.read_csv(f)
        df['Id'] = os.path.basename(f).replace('.csv', '')
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df[(df['Valid'] == True) & (df['Task'] == True)]


def load_tdcsfog() -> pd.DataFrame:
    dfs = []
    for f in glob.glob(f'{TDCSFOG_DIR}*.csv'):
        df = pd.read_csv(f)
        df['Id'] = os.path.basename(f).replace('.csv', '')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def normalize_sessions(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for _, group in df.groupby('Id'):
        group = group.copy()
        group[FEATURES] = StandardScaler().fit_transform(group[FEATURES])
        result.append(group)
    return pd.concat(result, ignore_index=True)


def make_windows(df: pd.DataFrame):
    X, y = [], []
    for _, group in df.groupby('Id'):
        group  = group.sort_values('Time').reset_index(drop=True)
        feats  = group[FEATURES].values
        labels = group[LABELS].values
        for start in range(0, len(group) - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            X.append(feats[start:end])
            y.append(labels[start:end].max(axis=0))
    return np.array(X), np.array(y)


def build_dataloaders():
    df_defog   = normalize_sessions(load_defog())
    df_tdcsfog = normalize_sessions(load_tdcsfog())

    X1, y1 = make_windows(df_defog)
    X2, y2 = make_windows(df_tdcsfog)

    X_all = np.concatenate([X1, X2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)
    print(f"Dataset: {X_all.shape}  |  FOG rate: {y_all.mean(axis=0)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    def to_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return to_loader(X_train, y_train, shuffle=True), to_loader(X_val, y_val), y_train