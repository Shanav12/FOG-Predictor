import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from data import build_dataloaders
from models import FOGLSTMModel, FOGRNNModel, FOG1DCNNModel
from config import (
    DEVICE, EPOCHS, LEARNING_RATE, LABELS,
    LSTM_MODEL_PATH, RNN_MODEL_PATH, CNN_MODEL_PATH, LSTM_FOCAL_MODEL_PATH,
    FOCAL_ALPHA, FOCAL_GAMMA,
)
from sklearn.metrics import (
    average_precision_score,
    precision_score, recall_score, f1_score,
    PrecisionRecallDisplay,
)



def build_pos_weight_criterion(y_train: np.ndarray) -> nn.BCEWithLogitsLoss:
    pos_weight = torch.tensor([(y_train[:, i] == 0).sum() / max((y_train[:, i] == 1).sum(), 1) 
                               for i in range(y_train.shape[1])],
        dtype=torch.float32,
    ).to(DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight = pos_weight)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float=FOCAL_ALPHA, gamma: float=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t  = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


def train(model, train_loader, val_loader, criterion, save_path: str, epochs: int=EPOCHS) -> None:
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch}/{epochs} | "f"training loss: {avg_train:.4f} | validation loss: {avg_val:.4f}")

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f"Best validation loss: {best_val:.4f}")


def collect_preds(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            all_preds.append(p)
            all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)



def evaluate(model, val_loader, model_name: str='model', threshold: float=0.5, plot: bool=True) -> tuple:
    preds, labels = collect_preds(model, val_loader)
    binary = (preds >= threshold).astype(int)
    aps = []
    for i, name in enumerate(LABELS):
        ap = average_precision_score(labels[:, i], preds[:, i])
        p  = precision_score(labels[:, i], binary[:, i], zero_division=0)
        r  = recall_score(labels[:, i], binary[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], binary[:, i], zero_division=0)
        aps.append(ap)
        print(f"{name:<20} {ap:>6.4f} {p:>6.4f} {r:>6.4f} {f1:>6.4f}")

    map_score = average_precision_score(labels, preds, average='macro')
    print(f"MAP Score: {map_score}")
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f'Precision-Recall Curve for {model_name}', fontsize=13)
        for i, (name, ax) in enumerate(zip(LABELS, axes)):
            PrecisionRecallDisplay.from_predictions(
                labels[:, i], preds[:, i], name=name, ax=ax)
            ax.set_title(f'{name}  AP={aps[i]:.3f}')
            ax.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        fname = f'pr_curve_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.show()

    return preds, labels


def error_analysis(preds: np.ndarray, labels: np.ndarray, model_name: str='model', threshold: float=0.5) -> None:
    binary = (preds >= threshold).astype(int)
    for i, name in enumerate(LABELS):
        pos = labels[:, i] == 1
        neg = labels[:, i] == 0
        tp = int(((binary[:, i] == 1) & pos).sum())
        fp = int(((binary[:, i] == 1) & neg).sum())
        fn = int(((binary[:, i] == 0) & pos).sum())
        tot = int(pos.sum())
        print(f"""
              {name:<20} total={tot:>5}\n
              TP={tp:>4}  FP={fp:>5}  FN={fn:>4}\n
              f"miss={fn/max(tot,1):.1%}
        """)


def tune_thresholds(preds: np.ndarray, labels: np.ndarray, model_name: str='model') -> list:
    best_thresholds = []
    for i, _ in enumerate(LABELS):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(labels[:, i], (preds[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best_thresholds.append(best_t)
        print(f"Best threshold: {best_t:.2f}  F1={best_f1:.4f}")
    return best_thresholds


def print_summary(results: dict) -> None:
    rows = []
    for name, (preds, labels) in results.items():
        binary = (preds >= 0.5).astype(int)
        rows.append({
            'Model': name,
            'MAP': round(average_precision_score(labels, preds, average='macro'), 4),
            'macro-F1': round(f1_score(labels, binary, average='macro', zero_division=0), 4),
            'AP_StartHesitation': round(average_precision_score(labels[:, 0], preds[:, 0]), 4),
            'AP_Turn': round(average_precision_score(labels[:, 1], preds[:, 1]), 4),
            'AP_Walking': round(average_precision_score(labels[:, 2], preds[:, 2]), 4),
        })
    df = pd.DataFrame(rows).set_index('Model')
    print(df.to_string())


def main():
    train_loader, val_loader, y_train = build_dataloaders()
    bce_criterion   = build_pos_weight_criterion(y_train)
    focal_criterion = FocalLoss()
    results = {}

    rnn = FOGRNNModel()
    train(rnn, train_loader, val_loader, bce_criterion, RNN_MODEL_PATH)
    preds, labels = evaluate(rnn, val_loader, 'RNN')
    error_analysis(preds, labels, 'RNN')
    results['RNN (baseline)'] = (preds, labels)
    lstm = FOGLSTMModel()
    train(lstm, train_loader, val_loader, bce_criterion, LSTM_MODEL_PATH)
    preds, labels = evaluate(lstm, val_loader, 'LSTM + pos_weight')
    error_analysis(preds, labels, 'LSTM + pos_weight')
    tune_thresholds(preds, labels, 'LSTM + pos_weight')
    results['LSTM + pos_weight'] = (preds, labels)

    
    cnn = FOG1DCNNModel()
    train(cnn, train_loader, val_loader, bce_criterion, CNN_MODEL_PATH)
    preds, labels = evaluate(cnn, val_loader, '1D-CNN')
    error_analysis(preds, labels, '1D-CNN')
    results['1D-CNN'] = (preds, labels)

    
    lstm_focal = FOGLSTMModel()
    train(lstm_focal, train_loader, val_loader, focal_criterion, LSTM_FOCAL_MODEL_PATH)
    preds, labels = evaluate(lstm_focal, val_loader, 'LSTM + Focal Loss')
    error_analysis(preds, labels, 'LSTM + Focal Loss')
    results['LSTM + Focal Loss'] = (preds, labels)

    print_summary(results)


if __name__ == '__main__':
    main()