"""Frozen original CF scores on every B frame, with archived endpoint controls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import types


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    root=args.run.resolve();manifest=load(root/'manifest.json')
    smoacs=Path(__file__).resolve().parents[2]/'smoacs'
    sys.path.insert(0,str(smoacs))
    from run_section21 import SOURCE_MEMBER,SOURCE_SHA256
    assets=Path('results/smoacs/assets-199').resolve();archive=Path('results/smoacs/downloads/reproducibility.tar.gz')
    if sha(archive)!=load(smoacs/'assets.json')['reproducibility.tar.gz']['sha256']:raise ValueError('Author archive mismatch.')
    checkpoints=load(smoacs/'checkpoints.json')
    for name,item in checkpoints.items():
        if sha(assets/'models/crystalformer'/name)!=item['sha256']:raise ValueError('Frozen CF checkpoint changed.')
    with tarfile.open(archive) as handle:original=handle.extractfile(SOURCE_MEMBER).read()
    if hashlib.sha256(original).hexdigest()!=SOURCE_SHA256:raise ValueError('Author source mismatch.')
    os.environ.update(SMOACS_ROOT=str(assets),MPLBACKEND='Agg')
    source=Path('third_party/SMOACS').resolve();sys.path[:0]=[str(source),str(source/'src')]
    author=types.ModuleType('relaxation_path_author');author.__file__=str(assets/SOURCE_MEMBER)
    sys.modules[author.__name__]=author
    exec(compile(original,author.__file__,'exec',dont_inherit=True),author.__dict__)
    import torch
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU CF required.')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models=author.Load_Pretrained_Crystalformers(device='cpu')
    for model in models:
        model.to('cuda').eval()
        for p in model.parameters():p.requires_grad_(False)
        if not all(p.is_cuda for p in model.parameters()):raise ValueError('CF CPU fallback.')
    matcher=StructureMatcher(ltol=.002,stol=.005,angle_tol=.2,primitive_cell=False,scale=False,attempt_supercell=False)
    all_rows=[];controls=[]
    for task in manifest['cases']:
        folder=Path(task['output']);result=load(folder/'result.json')
        frames=[json.loads(x) for x in (folder/'trajectory/frames.jsonl').read_text().splitlines()]
        if frames[0]['stage']!='input' or frames[1]['B_step']!=0:raise ValueError('Input or post-symmetry initial frame missing.')
        scores=[]
        for row in frames:
            path=Path(row['path'])
            if sha(path)!=row['sha256']:raise ValueError('Observed pose changed.')
            structure=Structure.from_file(path);n=len(structure)
            cell=torch.tensor(structure.lattice.matrix,dtype=torch.float32,device='cuda')
            atomic=torch.nn.functional.one_hot(torch.tensor(structure.atomic_numbers,device='cuda')-1,num_classes=98).float()
            coords=torch.tensor(structure.frac_coords,dtype=torch.float32,device='cuda')@cell
            with torch.no_grad():
                gap,ef=author.Crystalformer_prediction(*models,atomic,coords,torch.zeros(n,dtype=torch.long,device='cuda'),
                    cell[None],torch.tensor([n],dtype=torch.long,device='cuda'))
            if not gap.is_cuda or not ef.is_cuda:raise ValueError('CF output not GPU.')
            gap,ef=float(gap.reshape(-1)[0]),float(ef.reshape(-1)[0])
            if not math.isfinite(gap) or not math.isfinite(ef):raise ValueError('Nonfinite CF score.')
            scored=dict(row,arm=task['arm'],candidate_id=task['candidate_id'],CF_gap=gap,CF_eform=ef,gap_absolute_error=abs(gap-2.5))
            scores.append(scored);all_rows.append(scored)
        initial_error=max(abs(scores[0]['CF_gap']-task['original_before']['crystalformer_gap']),abs(scores[0]['CF_eform']-task['original_before']['crystalformer_eform']))
        final_error=max(abs(scores[-1]['CF_gap']-task['original_after']['crystalformer_gap']),abs(scores[-1]['CF_eform']-task['original_after']['crystalformer_eform']))
        initial_pass=initial_error<=1e-3
        terminal_match=matcher.fit(Structure.from_file(folder/'relaxed.vasp'),Structure.from_file(task['original_result']['relaxed_path']))
        energy_error=abs(result['final_energy_eV']-task['original_result']['final_energy_eV'])/scores[-1]['num_atoms']
        passed=initial_pass and final_error<=1e-3 and terminal_match and energy_error<=1e-4 and result['status']=='converged'
        controls.append(dict(arm=task['arm'],candidate_id=task['candidate_id'],initial_prediction_error=initial_error,
            final_prediction_error=final_error,terminal_structure_match=bool(terminal_match),terminal_energy_error_eV_atom=energy_error,
            initial_passed=initial_pass,passed=passed,optimizer_steps=result['optimizer_steps'],frames=len(frames)))
        print(json.dumps(controls[-1]),flush=True)
    report=dict(passed=all(r['passed'] for r in controls),initial_controls_passed=all(r['initial_passed'] for r in controls),
        controls=controls,CF_source_sha256=SOURCE_SHA256,checkpoints=checkpoints,torch=torch.__version__,device=torch.cuda.get_device_name(),
        interpretation='Frozen saved-frame CF predictions, not DFT; replay equivalence is separately checked against archived inputs and final results.')
    (root/'scores.json').write_text(json.dumps(all_rows,indent=2)+'\n')
    (root/'replay-controls.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['initial_controls_passed']:raise ValueError('Frozen CF source-input equivalence failed.')


if __name__=='__main__':main()
