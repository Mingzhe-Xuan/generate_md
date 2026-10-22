"""Asset preparation tests use tiny fixtures and never run model computation."""
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

SCRIPT = Path(__file__).with_name("prepare_assets.py")
spec = importlib.util.spec_from_file_location("smoacs_prepare", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("name", [
    "../outside", "/absolute",
    "reproducibility/section_2_1/templates_masks/../source.py",
    "reproducibility/section_2_1/templates_masks/bad\\name",
])
def test_rejects_unsafe_archive_paths(name):
    with pytest.raises(ValueError):
        module.data_path(name)


def fixture_archives(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    with tarfile.open(downloads / "reproducibility.tar.gz", "w:gz") as archive:
        for name, data in {
            "reproducibility/section_2_1/templates_masks/example.vasp": b"template",
            "reproducibility/section_2_1/run_section_2_1.sh": b"source must stay out",
        }.items():
            header = tarfile.TarInfo(name)
            header.size = len(data)
            archive.addfile(header, io.BytesIO(data))
    with zipfile.ZipFile(downloads / "crystalformer_weights.zip", "w") as archive:
        for task in ("bandgap", "e_form"):
            archive.writestr(f"weights/megnet-{task}-layer4.ckpt", b"fixture")
    (downloads / "crystalformer_default.json").write_bytes(b"{}")
    manifest = {
        path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size}
        for path in downloads.iterdir()
    }
    return downloads, manifest


def test_extracts_only_data_and_required_weights(tmp_path):
    downloads, manifest = fixture_archives(tmp_path)
    output = tmp_path / "prepared"
    result = module.prepare(downloads, output, manifest)
    assert result["data_files"] == 1
    assert (output / "reproducibility/section_2_1/templates_masks/example.vasp").read_bytes() == b"template"
    assert not (output / "reproducibility/section_2_1/run_section_2_1.sh").exists()
    assert (output / "models/crystalformer/default.json").read_bytes() == b"{}"
    assert (output / "prepared.json").is_file()
    with pytest.raises(FileExistsError):
        module.prepare(downloads, output, manifest)
    assert (output / "prepared.json").is_file()


def test_rejects_corrupt_input_before_creating_output(tmp_path):
    downloads, manifest = fixture_archives(tmp_path)
    (downloads / "crystalformer_default.json").write_bytes(b"!!")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="SHA-256"):
        module.prepare(downloads, output, manifest)
    assert not output.exists()


def test_cli_refuses_bulk_work_without_slurm(tmp_path):
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--downloads", str(tmp_path), "--output", str(output)],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "through Slurm" in result.stderr
    assert not output.exists()


def direct_fixture(downloads):
    direct = {}
    for name in ("megnet-bandgap-layer4.ckpt", "megnet-e_form-layer4.ckpt"):
        content = b"fixture"
        (downloads / name).write_bytes(content)
        direct[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    return direct


def test_direct_checkpoint_mode_preserves_the_same_model_bytes(tmp_path):
    downloads, manifest = fixture_archives(tmp_path)
    direct = direct_fixture(downloads)
    (downloads / "crystalformer_weights.zip").unlink()
    output = tmp_path / "prepared"
    result = module.prepare(downloads, output, manifest, direct)
    assert result["weights_source"] == "direct_checkpoints"
    assert "crystalformer_weights.zip" not in result["verified_inputs"]
    for name in direct:
        assert (output / "models/crystalformer" / name).read_bytes() == b"fixture"


def test_direct_mode_rejects_altered_checkpoint_before_output(tmp_path):
    downloads, manifest = fixture_archives(tmp_path)
    direct = direct_fixture(downloads)
    (downloads / "megnet-bandgap-layer4.ckpt").write_bytes(b"changed")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        module.prepare(downloads, output, manifest, direct)
    assert not output.exists()
