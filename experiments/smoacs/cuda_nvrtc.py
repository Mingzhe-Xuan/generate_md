"""Select NVRTC directly for original Crystalformer CUDA sources, without Jitify."""
from __future__ import annotations
from pathlib import Path

class NvrtcCompiler:
    def __init__(self, cupy_module, include_root: Path):
        self.cupy = cupy_module
        self.include_root = include_root.resolve()

    def __getattr__(self, name):
        return getattr(self.cupy, name)

    def RawKernel(self, code, name, options=(), **kwargs):
        kwargs["jitify"] = False
        options = tuple(options) + ("-I" + str(self.include_root),)
        return self.cupy.RawKernel(code, name, options, **kwargs)

def enable_plain_nvrtc():
    from crystalformer.models.cuda_funcs import kernel_manager
    root = Path(kernel_manager.__file__).resolve().parents[2]
    if not (root / "models/kernels").is_dir():
        raise RuntimeError("Cannot locate original Crystalformer kernel headers.")
    if not isinstance(kernel_manager.cp, NvrtcCompiler):
        kernel_manager.cp = NvrtcCompiler(kernel_manager.cp, root)
    return {"compiler": "nvrtc", "jitify": False, "include_root": str(root)}
