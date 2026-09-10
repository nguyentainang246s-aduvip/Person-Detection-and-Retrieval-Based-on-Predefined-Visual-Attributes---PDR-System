"""
src/utils/onnx_engine.py
========================
High-performance ONNX Runtime inference engine for CPU/Edge deployment.
Supports graph optimization, intra/inter thread tuning, and batch inference.
"""

import os
import logging
from typing import List, Union
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXInferenceEngine:
    """
    Wrapper quanh onnxruntime.InferenceSession tối ưu hóa tính toán ma trận trên CPU/GPU.
    """

    def __init__(
        self,
        onnx_model_path: str,
        device: str = "cpu",
        num_threads: int = None
    ):
        self.model_path = onnx_model_path
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình ONNX: {onnx_model_path}")

        # Cấu hình tối ưu đồ thị tính toán
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Cấu hình số luồng song song
        if num_threads:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads

        # Thiết lập Execution Providers
        providers = ["CPUExecutionProvider"]
        if device.lower() in ["cuda", "gpu"] and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape

    def run(self, input_tensor: Union[np.ndarray, "torch.Tensor"]) -> List[np.ndarray]:
        """
        Chạy suy luận trên tensor đầu vào. Chấp nhận cả NumPy array lẫn PyTorch Tensor.
        """
        if hasattr(input_tensor, "detach"):
            input_data = input_tensor.detach().cpu().numpy()
        else:
            input_data = np.asarray(input_tensor, dtype=np.float32)

        # Chạy suy luận qua ONNX Runtime C++ engine
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: input_data}
        )
        return outputs

    def __call__(self, input_tensor):
        return self.run(input_tensor)

    @property
    def file_size_mb(self) -> float:
        """Kích thước file mô hình (MB)."""
        return os.path.getsize(self.model_path) / (1024 * 1024)
