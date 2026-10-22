"""Verify and extract downloaded paper data inside Slurm, without installing source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile


def data_path(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    if (len(path.parts) >= 4 and path.parts[0] == "reproducibility"
            and path.parts[1] in {"section_2_1", "section_2_2", "section_2_3", "section_2_4"}
            and path.parts[2] == "templates_masks"):
        return path
    return None


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare(downloads: Path, output: Path, manifest: dict, checkpoints: dict | None = None) -> dict:
    verification = {name: spec for name, spec in manifest.items()
                    if checkpoints is None or name != "crystalformer_weights.zip"}
    verification.update(checkpoints or {})
    for name, spec in verification.items():
        path = downloads / name
        if "bytes" in spec and path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        if digest(path) != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    files = 0
    total_bytes = 0
    with tarfile.open(downloads / "reproducibility.tar.gz", "r:gz") as archive:
        for member in archive:
            relative = data_path(member.name)
            if relative is None or member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Refusing non-regular data: {member.name}")
            target = output.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            files += 1
            total_bytes += member.size
    model_dir = output / "models/crystalformer"
    model_dir.mkdir(parents=True)
    names = ["megnet-bandgap-layer4.ckpt", "megnet-e_form-layer4.ckpt"]
    model_hashes = {}
    if checkpoints is None:
        with zipfile.ZipFile(downloads / "crystalformer_weights.zip") as archive:
            for name in names:
                with archive.open("weights/" + name) as source, (model_dir / name).open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    else:
        for name in names:
            shutil.copyfile(downloads / name, model_dir / name)
    for name in names:
        model_hashes[name] = digest(model_dir / name)
    shutil.copyfile(downloads / "crystalformer_default.json", model_dir / "default.json")
    if files == 0:
        raise ValueError("Archive contained no selected template data.")
    report = {"job_id": os.environ.get("SLURM_JOB_ID"), "data_files": files,
              "data_bytes": total_bytes, "checkpoint_sha256": model_hashes,
              "input_manifest": manifest, "verified_inputs": verification,
              "weights_source": "original_zip" if checkpoints is None else "direct_checkpoints",
              "source_extracted": False}
    (output / "prepared.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-checkpoints", action="store_true",
                        help="Verify and use the two unchanged layer-4 files instead of the full ZIP.")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run bulk preparation through Slurm.")
    allowed = (Path.cwd() / "results/smoacs").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        parser.error("Output must be a new child directory under results/smoacs.")
    manifest = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))
    checkpoints = None
    if args.direct_checkpoints:
        checkpoints = json.loads(Path(__file__).with_name("checkpoints.json").read_text(encoding="utf-8"))
    report = prepare(args.downloads.resolve(), output, manifest, checkpoints)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
