import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from config import DEVICE, EPOCHS, LEARNING_RATE, LSTM_MODEL_PATH, RNN_MODEL_PATH, LABELS
from data import build_dataloaders
from models import FOGLSTMModel, FOGRNNModel



def build_criterion(y_train: np.ndarray) -> nn.BCEWithLogitsLoss:
    pos_weight = torch.tensor(
        [(y_train[:, i] == 0).sum() / max((y_train[:, i] == 1).sum(), 1)
         for i in range(y_train.shape[1])],
        dtype=torch.float32,
    ).to(DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)



def train(model, train_loader, val_loader, criterion, epochs: int = EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(DEVICE)

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

        print(
            f"Epoch {epoch + 1:>2}/{epochs} | "
            f"train loss: {train_loss / len(train_loader):.4f} | "
            f"val loss:   {val_loss / len(val_loader):.4f}"
        )



def evaluate(model, val_loader, label_names=LABELS) -> float:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            preds = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    for i, name in enumerate(label_names):
        ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        print(f"  {name}: AP = {ap:.4f}")

    map_score = average_precision_score(all_labels, all_preds, average='macro')
    print(f"  MAP Score: {map_score:.4f}")
    return map_score



def main():
    print(DEVICE)
    train_loader, val_loader, y_train = build_dataloaders()
    criterion = build_criterion(y_train)
    print("\n=== Training LSTM ===")
    lstm_model = FOGLSTMModel()
    train(lstm_model, train_loader, val_loader, criterion)
    print("\nLSTM Evaluation:")
    evaluate(lstm_model, val_loader)
    torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
    print(f"Saved")

    print("\n=== Training RNN ===")
    rnn_model = FOGRNNModel()
    train(rnn_model, train_loader, val_loader, criterion)
    print("\nRNN Evaluation:")
    evaluate(rnn_model, val_loader)
    torch.save(rnn_model.state_dict(), RNN_MODEL_PATH)
    print(f"Saved")


if __name__ == '__main__':
    main()