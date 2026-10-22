"""Check the actual SMOACS checkout and paper-run assets without importing ML packages."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

PINNED_COMMIT = "24fe8467de87167818e75ccd7eec21ab673f8df5"


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def inspect(source: Path) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    try:
        commit = git(source, "rev-parse", "HEAD")
        remote = git(source, "remote", "get-url", "origin")
        check("pinned_commit", commit == PINNED_COMMIT, commit)
        check("origin", remote == "https://github.com/AkiraTOSEI/SMOACS.git", remote)
        check("upstream_clean", not git(source, "status", "--porcelain"), "git status --porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        check("git_checkout", False, str(exc))

    required = [
        "main.py", "LICENSE", "THIRD_PARTY_NOTICES.md",
        "data/intermidiate/mask_dict_super_common.npz",
        "data/intermidiate/ionic_mask_dict_super_common.npz",
        "data/intermidiate/ionic_radii_dict_super_common.npz",
        "reproducibility/experiment_manifest.md",
        "models/crystalformer/default.json",
        "models/crystalformer/megnet-bandgap-layer4.ckpt",
        "models/crystalformer/megnet-e_form-layer4.ckpt",
    ]
    for section in ("2_1", "2_2", "2_3", "2_4"):
        required.append(f"reproducibility/section_{section}/run_section_{section}.sh")
        inputs = source / f"reproducibility/section_{section}/templates_masks"
        check(f"section_{section}_inputs", inputs.is_dir() and any(inputs.iterdir()),
              str(inputs))
    for relative in required:
        path = source / relative
        check(relative, path.is_file() and path.stat().st_size > 0, str(path))

    manifest = source / "manifests/model_weights_manifest.tsv"
    if manifest.is_file():
        import csv
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        for row in rows:
            if row["artifact_id"] not in {"mp_gappbe_alignnn", "mp_e_form_alignnn"}:
                continue
            path = source / row["default_path"]
            actual = None
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual = digest.hexdigest()
            check(row["artifact_id"], actual == row["expected_sha256"],
                  f"expected={row['expected_sha256']}; actual={actual}")
    else:
        check("weight_manifest", False, str(manifest))

    versions = {}
    for package in ("torch", "dgl", "torch-geometric", "numpy", "pymatgen",
                    "periodictable", "smact", "ase"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "source": str(source), "checks": checks, "installed_versions": versions,
        "listed_assets_ready": all(item["ok"] for item in checks),
        "scope": "Source and asset checks only; no GPU, gradient or optimization validation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[2] / "third_party/SMOACS")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.source.resolve())
    report = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0 if result["listed_assets_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
