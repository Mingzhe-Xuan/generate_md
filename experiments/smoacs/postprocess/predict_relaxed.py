"""Measure relaxation drift using exactly the CPU206 archived predictors."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import types
from inventory import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Inference must run through Slurm.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_section21 import SOURCE_MEMBER, SOURCE_SHA256
    assets = args.assets.resolve()
    archive_path = Path('results/smoacs/downloads/reproducibility.tar.gz')
    manifest = json.loads(Path(__file__).resolve().parents[1].joinpath('assets.json').read_text())
    if sha256(archive_path) != manifest['reproducibility.tar.gz']['sha256']:
        raise ValueError('Author archive changed.')
    weights = json.loads(Path(__file__).resolve().parents[1].joinpath('checkpoints.json').read_text())
    for name, metadata in weights.items():
        if sha256(assets/'models/crystalformer'/name) != metadata['sha256']:
            raise ValueError('Original CPU predictor checkpoint changed.')
    with tarfile.open(archive_path) as archive:
        original = archive.extractfile(SOURCE_MEMBER).read()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError('Author source changed.')
    os.environ['SMOACS_ROOT'] = str(assets)
    os.environ['MPLBACKEND'] = 'Agg'
    source = Path('third_party/SMOACS').resolve()
    sys.path[:0] = [str(source), str(source/'src')]
    module = types.ModuleType('paper_cpu206_predictor')
    module.__file__ = str(assets/SOURCE_MEMBER)
    sys.modules[module.__name__] = module
    exec(compile(original, module.__file__, 'exec', dont_inherit=True), module.__dict__)
    import torch
    from pymatgen.core import Structure
    torch.set_num_threads(4)
    bandgap, eform = module.Load_Pretrained_Crystalformers(device='cpu')
    def predict(path):
        structure = Structure.from_file(path)
        n = len(structure)
        atomic = torch.nn.functional.one_hot(torch.tensor(structure.atomic_numbers)-1, num_classes=98).float()
        cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
        coords = torch.tensor(structure.frac_coords, dtype=torch.float32)@cell
        with torch.no_grad():
            gap, ef = module.Crystalformer_prediction(bandgap, eform, atomic, coords,
                torch.zeros(n,dtype=torch.long), cell[None,:,:], torch.tensor([n],dtype=torch.long))
        values = float(gap.reshape(-1)[0]), float(ef.reshape(-1)[0])
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Non-finite CPU predictor output.')
        return values
    candidates = json.loads(args.inventory.read_text())['candidates'][:args.limit]
    rows = []
    for candidate in candidates:
        if sha256(candidate['source']) != candidate['sha256']:
            raise ValueError('Original structure changed.')
        gap, ef = predict(candidate['source'])
        row = dict(candidate_id=candidate['candidate_id'], gap_original_csv=candidate['original_gap'],
                   eform_original_csv=candidate['original_eform'], gap_before=gap,eform_before=ef,
                   gap_control_error=abs(gap-candidate['original_gap']),
                   eform_control_error=abs(ef-candidate['original_eform']))
        record_path = args.root/'relaxation'/Path(candidate['candidate_id']).stem/'result.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            row['relaxation_status'] = record['status']
            if record['status'] in ['converged','unconverged']:
                if sha256(record['relaxed_path']) != record['relaxed_sha256']:
                    raise ValueError('Relaxed structure changed.')
                gap_after, ef_after = predict(record['relaxed_path'])
                row.update(gap_after=gap_after, eform_after=ef_after,
                           gap_drift=gap_after-gap, eform_drift=ef_after-ef,
                           relaxed_gap_target_success=abs(gap_after-2.5)<=0.2,
                           relaxed_eform_target_success=ef_after< -0.5)
        rows.append(row)
        print(json.dumps(row),flush=True)
    out = args.root/('predictor-'+os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    with (out/'predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)
    control_error=max(max(r['gap_control_error'],r['eform_control_error']) for r in rows)
    report=dict(candidate_count=len(rows), source_sha256=SOURCE_SHA256, device='cpu',
                max_original_control_error=control_error, control_tolerance=1e-3,
                control_passed=control_error<=1e-3,
                purpose='same-predictor relaxation drift; not DFT or stability validation')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['control_passed']:
        raise RuntimeError('Original POSCAR control differs from CPU206 CSV beyond tolerance.')


if __name__ == '__main__':
    main()
