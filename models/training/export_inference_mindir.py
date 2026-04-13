import os
import sys
from typing import Optional

import numpy as np
import mindspore as ms
from mindspore import Tensor, context, export, load_checkpoint, load_param_into_net


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.training.dataset_loader import GRUClassifier, MODEL_DIR


def export_inference_mindir(
    ckpt_path: Optional[str] = None,
    output_prefix: Optional[str] = None,
    batch_size: int = 1,
    sequence_length: int = 41,
    feature_dim: int = 153,
):
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    resolved_ckpt = ckpt_path or os.path.join(MODEL_DIR, 'version3.ckpt')
    resolved_output_prefix = output_prefix or os.path.join(MODEL_DIR, 'version3_infer')

    if not os.path.exists(resolved_ckpt):
        raise FileNotFoundError(f'未找到 checkpoint: {resolved_ckpt}')

    net = GRUClassifier(dropout=0.0)
    params = load_checkpoint(resolved_ckpt)
    load_result = load_param_into_net(net, params)
    if isinstance(load_result, tuple):
        not_loaded, _ = load_result
        if not_loaded:
            raise RuntimeError(f'部分参数未能加载到网络中: {not_loaded}')

    net.set_train(False)

    sample = Tensor(
        np.zeros((batch_size, sequence_length, feature_dim), dtype=np.float32),
        ms.float32,
    )
    export(net, sample, file_name=resolved_output_prefix, file_format='MINDIR')
    print(f'Saved inference mindir: {resolved_output_prefix}.mindir')


if __name__ == '__main__':
    export_inference_mindir()