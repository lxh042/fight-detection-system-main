import os
import sys

import numpy as np
import onnxruntime as ort
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.training.train_torch_onnx import MODEL_DIR, TorchGRUClassifier


def main():
    state_dict_path = os.path.join(MODEL_DIR, 'version3_torch.pt')
    onnx_path = os.path.join(MODEL_DIR, 'version3.onnx')

    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f'missing torch checkpoint: {state_dict_path}')
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f'missing onnx model: {onnx_path}')

    sample = np.random.randn(1, 41, 153).astype(np.float32)

    model = TorchGRUClassifier()
    model.load_state_dict(torch.load(state_dict_path, map_location='cpu', weights_only=True))
    model.eval()

    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).cpu().numpy().reshape(-1)[0]

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_out = session.run(['prob'], {'input': sample})[0].reshape(-1)[0]

    diff = abs(float(torch_out) - float(onnx_out))
    print(f'torch: {float(torch_out):.8f}')
    print(f'onnx : {float(onnx_out):.8f}')
    print(f'diff : {diff:.8f}')


if __name__ == '__main__':
    main()