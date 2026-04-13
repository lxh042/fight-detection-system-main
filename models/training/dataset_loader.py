import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# 优先用缓存的特征，如缺失则回退到按需生成
def _find_npz(mode: str):
    candidates = [
        os.path.join(PROJECT_ROOT, "preprocessed", f"{mode}_features_v1.npz"),
        os.path.join(CURRENT_DIR, "preprocessed", f"{mode}_features_v1.npz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def _load_or_prepare(mode: str, dataset_path: str):
    cache = _find_npz(mode)
    if cache:
        data = __import__('numpy').load(cache, allow_pickle=False)
        return data['X'], data['y']
    else:
        # 回退：生成并保存到 PROJECT_ROOT/preprocessed 下
        from utills.data_preprocessing import prepare_features
        X, y = prepare_features(dataset_path, mode, cache_dir=os.path.join(PROJECT_ROOT, "preprocessed"))
        return X, y

# Run two times one for test and one for train
featuresTest, labelsTest = _load_or_prepare("test", DATASET_PATH)
featuresTrain, labelsTrain = _load_or_prepare("train", DATASET_PATH)


# 新增：MindSpore 训练（GRU）+ tqdm 进度条 + 指标
import time
import numpy as np
from tqdm.auto import tqdm
import mindspore as ms
from mindspore import Tensor, context, nn, save_checkpoint, export
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# MindSpore 运行环境
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class GRUClassifier(nn.Cell):
    def __init__(self, input_size=153, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        # batch_first=True: 输入(B, T, F)
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          has_bias=True,
                          batch_first=True,
                          dropout=dropout)
        self.head = nn.SequentialCell(
            nn.Dense(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Dense(64, 1),
            nn.Sigmoid()
        )

    def construct(self, x):
        # x: (B, T, F)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # 取最后一个时间步
        return self.head(last)  # (B, 1)

def make_batches(X, y, batch_size, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, len(X), batch_size):
        e = min(s + batch_size, len(X))
        b = idx[s:e]
        yield X[b], y[b]

def evaluate(net: nn.Cell, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
    net.set_train(False)
    probs = []
    for bx, _ in make_batches(X, y, batch_size, shuffle=False):
        out = net(Tensor(bx.astype(np.float32)))
        probs.append(out.asnumpy().reshape(-1))
    y_prob = np.concatenate(probs) if len(probs) else np.array([], dtype=np.float32)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y, y_pred) if len(y) else 0.0
    rec = recall_score(y, y_pred, zero_division=0) if len(y) else 0.0
    f1  = f1_score(y, y_pred, zero_division=0) if len(y) else 0.0
    try:
        auc = roc_auc_score(y, y_prob) if len(y) else 0.0
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y, y_pred) if len(y) else np.zeros((2,2), dtype=int)
    return acc, rec, f1, auc, cm, y_prob

def train():
    # 数据
    X_train = featuresTrain.astype(np.float32)
    y_train = labelsTrain.astype(np.int32)
    X_val   = featuresTest.astype(np.float32)
    y_val   = labelsTest.astype(np.int32)

    # 模型/损失/优化器
    net = GRUClassifier()
    loss_fn = nn.BCELoss(reduction='mean')
    net_with_loss = nn.WithLossCell(net, loss_fn)
    opt = nn.Adam(net.trainable_params(), learning_rate=1e-3)
    train_step = nn.TrainOneStepCell(net_with_loss, opt)
    train_step.set_train()

    epochs = 100
    batch_size = 64

    for epoch in range(epochs):
        t0 = time.time()
        running_loss, seen = 0.0, 0

        bar = tqdm(make_batches(X_train, y_train, batch_size, shuffle=True),
                   total=(len(X_train) + batch_size - 1) // batch_size,
                   ncols=100, desc=f"Epoch {epoch+1}/{epochs}")
        for bx, by in bar:
            bx = bx.astype(np.float32)
            by = by.astype(np.float32).reshape(-1, 1)
            loss = train_step(Tensor(bx), Tensor(by))
            batch_loss = float(loss.asnumpy())
            running_loss += batch_loss * len(bx)
            seen += len(bx)

            # 当前学习率显示（兼容不同版本）
            try:
                lr_val = opt.get_lr()
                if isinstance(lr_val, Tensor):
                    lr_val = float(lr_val.asnumpy())
                elif hasattr(lr_val, "__array__"):
                    lr_val = float(np.array(lr_val).squeeze())
                else:
                    lr_val = float(lr_val)
            except Exception:
                lr_val = 1e-3

            bar.set_postfix(loss=f"{batch_loss:.4f}",
                            lr=f"{lr_val:.3e}",
                            time=f"{(time.time()-t0):.1f}s")

        epoch_loss = running_loss / max(1, seen)

        # 每轮评估
        acc, rec, f1, auc, cm, _ = evaluate(net, X_val, y_val)
        print(f"Epoch {epoch+1}/{epochs} | loss: {epoch_loss:.4f} | acc: {acc:.4f} | recall: {rec:.4f} | f1: {f1:.4f} | auc: {auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    # 最终评估与保存
    acc, rec, f1, auc, cm, _ = evaluate(net, X_val, y_val)
    print("Validation summary:")
    print(f"acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}, auc: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    ckpt_path = os.path.join(MODEL_DIR, "version3.ckpt")
    save_checkpoint(net, ckpt_path)

    mindir_prefix = os.path.join(MODEL_DIR, "version3")
    sample = Tensor(np.zeros((1, 41, 153), dtype=np.float32))
    export(net, sample, file_name=mindir_prefix, file_format="MINDIR")
    print(f"Saved: {ckpt_path}")
    print(f"Saved: {mindir_prefix}.mindir")

if __name__ == "__main__":
    train()