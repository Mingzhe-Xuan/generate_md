"""Compile representative unchanged Crystalformer CUDA kernels inside Slurm."""
import argparse
import json
import os
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Run through Slurm.")
    sys.path.insert(0, str(args.source.resolve() / "src"))
    from cuda_nvrtc import enable_plain_nvrtc
    report = enable_plain_nvrtc()
    import torch
    from crystalformer.utils import Params
    from crystalformer.models.latticeformer import Latticeformer
    from crystalformer.models.cuda_funcs import KernelManager
    model = Latticeformer(Params(str(args.params)))
    if type(model.encoder).__name__ != "CrystalformerEncoderCUDA":
        raise RuntimeError("CUDA encoder required.")
    names = ["real_enc_fwd_v2", "real_enc_bwd_v2", "fused_dpa_fwd_v3", "fused_dpa_bwd_v3"]
    for name in names:
        KernelManager.get_kernel(name).raw_kernel.compile()
        print("COMPILED " + name, flush=True)
    report.update(gpu=torch.cuda.get_device_name(0), kernels=names, passed=True,
                  scope="Compilation only, not a model forward/backward smoke.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
