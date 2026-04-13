import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score, recall_score, f1_score

# 解析工程根路径并导入预处理
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def load_test_features():
    candidates = [
        os.path.join(PROJECT_ROOT, "preprocessed", "test_features_v1.npz"),
        os.path.join(PROJECT_ROOT, "models", "training", "preprocessed", "test_features_v1.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=False)
            return data["X"], data["y"]

    return prepare_features(os.path.join(PROJECT_ROOT, "data"), "test")

from utills.data_preprocessing import prepare_features

import mindspore as ms
from mindspore import Tensor, context, nn
from mindspore.train.serialization import load

# MindSpore 环境
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

# 加载测试集特征
x_test, y_test = load_test_features()

# 加载 MindIR
mindir_path = os.path.join(MODEL_DIR, "version3.mindir")
graph = load(mindir_path)
model = nn.GraphCell(graph)

# 推理
# 当前导出的 MindIR 使用固定样例形状导出，评估时按单样本推理可避免批量维度不匹配。
probs = []
for i in range(len(x_test)):
    sample = x_test[i:i+1].astype(np.float32)
    out = model(Tensor(sample, ms.float32))
    probs.append(float(out.asnumpy().reshape(-1)[0]))
y_prob = np.array(probs, dtype=np.float32)
y_pred = (y_prob > 0.5).astype(int)

# 指标
acc = accuracy_score(y_test, y_pred) if len(y_test) else 0.0
rec = recall_score(y_test, y_pred, zero_division=0) if len(y_test) else 0.0
f1  = f1_score(y_test, y_pred, zero_division=0) if len(y_test) else 0.0
try:
    auc = roc_auc_score(y_test, y_prob) if len(y_test) else 0.0
except Exception:
    auc = float('nan')
cm = confusion_matrix(y_test, y_pred) if len(y_test) else np.zeros((2,2), dtype=int)

print(f"acc: {acc:.4f}, recall: {rec:.4f}, f1: {f1:.4f}, auc: {auc:.4f}")
print(f"Confusion Matrix:\n{cm}")

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(cm, display_labels=['non-violence','violence'])
disp.plot()
plt.show()