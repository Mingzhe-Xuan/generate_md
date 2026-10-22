"""Validate runtime imports and gradients inside a Slurm allocation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cupy", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run this compute probe through Slurm.")
    results = []

    def run(name, operation):
        try:
            detail = operation()
            results.append({"name": name, "passed": True, "detail": str(detail)})
        except Exception as exc:
            results.append({"name": name, "passed": False,
                            "detail": f"{type(exc).__name__}: {exc}"})

    def torch_probe(device):
        import torch
        x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        loss = x.square().sum()
        loss.backward()
        assert torch.allclose(x.grad, torch.tensor([2.0, 4.0], device=device))
        if device == "cuda":
            torch.cuda.synchronize()
            return {"gpu": torch.cuda.get_device_name(0),
                    "capability": torch.cuda.get_device_capability(0),
                    "built_architectures": torch.cuda.get_arch_list()}
        return "finite forward and correct gradient"

    def dgl_probe(device):
        import dgl
        import torch
        graph = dgl.graph(([0, 1], [1, 0])).to(device)
        graph.ndata["h"] = torch.ones(2, 1, device=device)
        return {"device": str(graph.device), "nodes": graph.num_nodes()}

    def cupy_probe():
        import cupy as cp
        from cupy.cuda import nvrtc
        runtime = cp.cuda.runtime.runtimeGetVersion()
        compiler = nvrtc.getVersion()
        assert runtime >= 12080 and compiler >= (12, 8), (runtime, compiler)
        data = cp.asarray([1.0, 2.0], dtype=cp.float32)
        kernel = cp.RawKernel(
            '#include <cuda_runtime.h>\nextern "C" __global__ void twice(float* x) { x[threadIdx.x] *= 2; }',
            "twice")
        kernel((1,), (2,), (data,))
        cp.cuda.get_current_stream().synchronize()
        assert cp.asnumpy(data).tolist() == [2.0, 4.0]
        return {"runtime": runtime, "nvrtc": compiler, "cuda_path": os.environ.get("CUDA_PATH")}

    def cli_probe():
        completed = subprocess.run(
            [sys.executable, str(args.source.resolve() / "main.py"), "--help"],
            cwd=args.source.resolve(), capture_output=True, text=True, timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(f"exit={completed.returncode}; {completed.stderr[-3000:]}")
        return "main.py --help exited 0"

    versions = {}
    for package in ("pip", "numpy", "torch", "dgl", "torch-geometric",
                    "pymatgen", "ase", "smact", "periodictable", "cupy-cuda12x", "pytorch-pfn-extras"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    run("torch_cpu_gradient", lambda: torch_probe("cpu"))
    run("dgl_cpu_graph", lambda: dgl_probe("cpu"))
    if args.device == "cuda":
        run("torch_cuda_gradient", lambda: torch_probe("cuda"))
        run("dgl_cuda_graph", lambda: dgl_probe("cuda"))
    if args.cupy:
        run("cupy_cuda_rawkernel", cupy_probe)
    run("main_cli_imports", cli_probe)
    report = {
        "job_id": os.environ["SLURM_JOB_ID"], "python": sys.version,
        "device_requested": args.device, "versions": versions,
        "checks": results, "passed": all(item["passed"] for item in results),
        "scope": "Runtime probe; no optimized structures or paper success rates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
