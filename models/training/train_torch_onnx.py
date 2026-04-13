import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')


def find_npz(mode: str) -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, 'preprocessed', f'{mode}_features_v1.npz'),
        os.path.join(CURRENT_DIR, 'preprocessed', f'{mode}_features_v1.npz'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'missing cached npz for {mode}: {candidates}')


def load_npz(mode: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(find_npz(mode), allow_pickle=False)
    return data['X'].astype(np.float32), data['y'].astype(np.float32)


class TorchGRUClassifier(nn.Module):
    def __init__(self, input_size: int = 153, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device):
    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
    preds = (probs > 0.5).astype(np.int32)
    acc = accuracy_score(y, preds)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)
    return acc, rec, f1, auc, cm, probs


def export_onnx(model: nn.Module, output_path: str):
    dummy = torch.randn(1, 41, 153, dtype=torch.float32)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['input'],
        output_names=['prob'],
        dynamic_axes={
            'input': {0: 'batch'},
            'prob': {0: 'batch'},
        },
        opset_version=13,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train a PyTorch GRU classifier and export it to ONNX.')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_train, y_train = load_npz('train')
    x_val, y_val = load_npz('test')

    device = torch.device('cpu')
    model = TorchGRUClassifier(
        input_size=x_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        model.train()
        indices = np.random.permutation(len(x_train))
        total_loss = 0.0

        for start in range(0, len(indices), args.batch_size):
            batch_ids = indices[start:start + args.batch_size]
            batch_x = torch.from_numpy(x_train[batch_ids]).to(device)
            batch_y = torch.from_numpy(y_train[batch_ids]).to(device).view(-1, 1)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_ids)

        acc, rec, f1, auc, cm, _ = evaluate(model, x_val, y_val, device)
        print(
            f'epoch={epoch + 1}/{args.epochs} '
            f'loss={total_loss / len(x_train):.4f} '
            f'acc={acc:.4f} recall={rec:.4f} f1={f1:.4f} auc={auc:.4f}'
        )
        print(cm)

    state_dict_path = os.path.join(MODEL_DIR, 'version3_torch.pt')
    onnx_path = os.path.join(MODEL_DIR, 'version3.onnx')
    torch.save(model.state_dict(), state_dict_path)
    export_onnx(model.cpu(), onnx_path)
    print(f'saved: {state_dict_path}')
    print(f'saved: {onnx_path}')


if __name__ == '__main__':
    main()