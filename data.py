import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from config import (
    DEFOG_DIR, TDCSFOG_DIR,
    FEATURES, LABELS,
    WINDOW_SIZE, STEP_SIZE,
    BATCH_SIZE, VAL_SUBJECTS, RANDOM_SEED,
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
    X, y, subjects = [], [], []
    for session_id, group in df.groupby('Id'):
        group = group.sort_values('Time').reset_index(drop=True)
        feats = group[FEATURES].values
        labels = group[LABELS].values
        for start in range(0, len(group) - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            X.append(feats[start:end])
            y.append(labels[start:end].max(axis=0))
            subjects.append(session_id)
    return np.array(X), np.array(y), np.array(subjects)


def subject_split(X: np.ndarray, y: np.ndarray, subjects: np.ndarray):    
    rng = np.random.default_rng(RANDOM_SEED)
    unique = np.unique(subjects)
    n_val  = max(1, int(VAL_SUBJECTS * len(unique)))
    val_ids = set(rng.choice(unique, size=n_val, replace=False))
    mask = np.array([s in val_ids for s in subjects])
    return X[~mask], X[mask], y[~mask], y[mask]


def to_dataloader(X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def build_dataloaders():
    df1 = normalize_sessions(load_defog())
    df2 = normalize_sessions(load_tdcsfog())

    X1, y1, s1 = make_windows(df1)
    X2, y2, s2 = make_windows(df2)

    X_all = np.concatenate([X1, X2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)
    s_all = np.concatenate([s1, s2], axis=0)

    X_train, X_val, y_train, y_val = subject_split(X_all, y_all, s_all)
    return (to_dataloader(X_train, y_train, shuffle=True), to_dataloader(X_val,   y_val), y_train)