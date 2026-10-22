"""Run the unchanged author Section 2.1 source from the verified archive, in Slurm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import types

SOURCE_MEMBER = "reproducibility/section_2_1/section_2_1_main.py"
SOURCE_SHA256 = "59f48b03c2066aae03a7526e72d3a6fc0a5539e850aa925cadb1aa43e453326d"


def execute_author_source(original: bytes, filename: str) -> None:
    module = types.ModuleType("__main__")
    module.__file__ = filename
    previous = sys.modules.get("__main__")
    sys.modules["__main__"] = module
    try:
        exec(compile(original, filename, "exec", dont_inherit=True), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--initialization", choices=[
        "random-random", "random-AtomicMask", "Struct-random", "Struct-AtomicMask",
    ], default="Struct-AtomicMask")
    parser.add_argument("--method", choices=["grad", "grad_sa", "mutation", "mutation_sa"],
                        default="grad")
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mutation-steps", type=int, nargs="+", default=[100, 150])
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run optimization through Slurm.")
    if Path(args.exp_name).name != args.exp_name or args.exp_name in {".", ".."}:
        parser.error("exp-name must be a single directory name.")
    if (args.candidates < 2 or args.candidates > 256 or args.batch_size < 2
            or args.batch_size > args.candidates or args.candidates % args.batch_size):
        parser.error("Require complete batches with 2 <= batch size <= candidates <= 256.")
    if args.steps < 2:
        parser.error("steps must be at least two.")
    output_root = args.output_root.resolve()
    allowed = (Path.cwd() / "results/smoacs").resolve()
    if not output_root.is_relative_to(allowed):
        parser.error("output-root must be under results/smoacs.")
    if (output_root / args.exp_name).exists():
        parser.error("Refusing to reuse an existing experiment directory.")
    source = args.source.resolve()
    assets = args.assets.resolve()
    prepared = json.loads((assets / "prepared.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))
    if prepared["input_manifest"] != manifest:
        raise ValueError("Prepared assets do not match the pinned manifest.")
    with args.archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
    if archive_hash is None:
        digest = hashlib.sha256()
        with args.archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        archive_hash = digest.hexdigest()
    if archive_hash != manifest["reproducibility.tar.gz"]["sha256"]:
        raise ValueError("Author archive SHA-256 mismatch.")
    with tarfile.open(args.archive, "r:gz") as archive:
        original = archive.extractfile(SOURCE_MEMBER).read()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Author source SHA-256 mismatch.")
    output_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "job_id": os.environ["SLURM_JOB_ID"], "author_source_sha256": SOURCE_SHA256,
        "archive_sha256": archive_hash, "arguments": vars(args),
        "algorithm_modified": False,
    }
    (output_root / (args.exp_name + "-launch.json")).write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8",
    )
    os.environ["SMOACS_ROOT"] = str(assets)
    os.environ["SMOACS_DEVICE"] = args.device
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "src"))
    synthetic_filename = str(assets / SOURCE_MEMBER)
    sys.argv = [
        synthetic_filename, "--initialization", args.initialization,
        "--method", args.method, "--num_candidate", str(args.candidates),
        "--b_size", str(args.batch_size), "--num_steps", str(args.steps),
        "--seed", str(args.seed), "--exp_name", args.exp_name,
        "--output_root", str(output_root), "--mutation_steps",
        *map(str, args.mutation_steps),
    ]
    execute_author_source(original, synthetic_filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
