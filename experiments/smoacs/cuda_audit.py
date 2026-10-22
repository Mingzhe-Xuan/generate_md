"""Observe real Crystalformer CUDA encoder forwards and their backward gradients."""
from __future__ import annotations
import json
import os
from pathlib import Path

class CudaKernelAudit:
    def __init__(self, output: Path):
        self.output = output
        self.report = {"encoder": "CrystalformerEncoderCUDA", "forward_calls": 0,
                       "backward_calls": 0, "passed": False}
        self.handle = None

    def __enter__(self):
        import torch
        if os.environ.get("SMOACS_CUDA_COMPILER", "upstream") == "nvrtc":
            from cuda_nvrtc import enable_plain_nvrtc
            self.report.update(enable_plain_nvrtc())
        from crystalformer.models.cuda_funcs import CUPY_AVAILABLE
        from crystalformer.models.indexed_latticeformer_encoder import CrystalformerEncoderCUDA
        if not CUPY_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError("CuPy CUDA implementation is required; refusing fallback.")
        self.report.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                           gpu=torch.cuda.get_device_name(0),
                           capability=list(torch.cuda.get_device_capability(0)))
        def backward(gradient):
            if not gradient.is_cuda or not torch.isfinite(gradient).all().item():
                raise RuntimeError("CUDA encoder gradient is not finite and on CUDA.")
            self.report["backward_calls"] += 1
        def forward(module, inputs, output):
            if isinstance(module, CrystalformerEncoderCUDA):
                if not isinstance(output, torch.Tensor) or not output.is_cuda:
                    raise RuntimeError("CUDA encoder output must be a CUDA tensor.")
                if not torch.isfinite(output).all().item():
                    raise RuntimeError("Non-finite CUDA encoder output.")
                self.report["forward_calls"] += 1
                if output.requires_grad:
                    output.register_hook(backward)
        self.handle = torch.nn.modules.module.register_module_forward_hook(forward)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.report["passed"] = (exc_type is None and self.report["forward_calls"] > 0
                                 and self.report["backward_calls"] > 0)
        if exc is not None:
            self.report["error"] = f"{type(exc).__name__}: {exc}"
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2)+"\n", encoding="utf-8")
        if exc_type is None and not self.report["passed"]:
            raise RuntimeError("No real Crystalformer CUDA forward/backward was observed.")
        return False
