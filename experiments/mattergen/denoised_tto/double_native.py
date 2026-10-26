"""Prove actual F64 proposal gradients and same-pose official F32 CF scoring."""
import argparse
import json
import os
from pathlib import Path
import traceback
import numpy as np
from adapter import Denoiser, sha
from mapping import LatentGeometry
from native import GradientScorer, promote, evaluate, save, POLICY

DOUBLE_POLICY = dict(parents=4, times=[.1, .3], seed=20261032, target_gap=2.5,
    proposal_precision='float64', latent_seed_arithmetic='float32_official_corruption',
    final_evaluator_precision='official_float32', geometry_arithmetic='difference-before-offset',
    same_pose_CF_tolerance=1e-5, source_CF_tolerance=1e-5,
    FD_steps=[1e-5, 1e-6], FD_absolute=.001, FD_relative=.02, search_trials=0,
    role='Separate actual-F64 execution proof; original F32-proxy gates remain failed.')


def case_passes(row):
    checks = row['checks']
    expected = {(b, e) for b in ('atomic', 'strain', 'both') for e in DOUBLE_POLICY['FD_steps']}
    return bool(len(checks) == 6 and {(q['block'], q['epsilon']) for q in checks} == expected
        and all(q.get('passed') is True for q in checks)
        and row['source_CF_error'] <= DOUBLE_POLICY['source_CF_tolerance']
        and row['same_pose_CF_error'] <= DOUBLE_POLICY['same_pose_CF_tolerance']
        and row['corrected_vs_native_forward'] <= DOUBLE_POLICY['same_pose_CF_tolerance']
        and np.isfinite(row['gradient']).all() and np.linalg.norm(row['gradient']) > 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm GPU.')
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = dict(policy=DOUBLE_POLICY, source=str(args.source.resolve()), cases=[], passed=False,
        original_float32_gate_reclassified=False, implementation_sha256={q.name: sha(q) for q in Path(__file__).parent.glob('*.py')})
    try:
        from ase import Atoms
        from ase.io import write
        from pymatgen.core import Structure
        plan = json.loads((args.source/'plan.json').read_text())
        if len(plan['cases']) != 16 or not all(q['neutral'] for q in plan['cases']):
            raise ValueError('Original neutral16 source required.')
        result['source_hashes'] = {n: sha(args.source/n) for n in ('plan.json', 'native-probe.json', 'direction-results.json', 'report.json')}
        if not json.loads((args.source/'native-probe.json').read_text())['passed']:
            raise ValueError('Source derivative proof failed.')
        denoiser = Denoiser(stable_offsets=True)
        scorer = GradientScorer()
        official = GradientScorer()
        torch = scorer.torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(2)
        result.update(device=torch.cuda.get_device_name(), MatterGen=denoiser.metadata, CF=scorer.metadata, official_CF=official.metadata)
        operators = []
        for case in plan['cases'][:4]:
            path = Path(case['B']['relaxed_path'])
            if sha(path) != case['B']['relaxed_sha256']:
                raise ValueError('Original pose changed.')
            s = Structure.from_file(path)
            gap, ef = official.predict(path)
            original = case['predictions']['after']
            error = max(abs(gap-original['crystalformer_gap']), abs(ef-original['crystalformer_eform']))
            if error > DOUBLE_POLICY['source_CF_tolerance']:
                raise ValueError('Source official CF control failed.')
            geometry = LatentGeometry(s.atomic_numbers, s.lattice.matrix, s.frac_coords)
            index = int(case['parent_id'].split('-')[1].split('.')[0])
            for time in DOUBLE_POLICY['times']:
                with torch.no_grad():
                    graph = denoiser.graph(geometry, torch.zeros(geometry.size, device='cuda'))
                    noisy, _ = denoiser.forward_noise(graph, time, [DOUBLE_POLICY['seed'], index, int(time*1000)])
                latent = LatentGeometry(geometry.numbers, noisy.cell[0].cpu().double().numpy(), noisy.pos.cpu().double().numpy())
                operators.append((case, time, latent, error))
        denoiser.module.double()
        promote(scorer, torch.float64)
        if any(q.dtype != torch.float64 for q in denoiser.module.parameters()) or any(q.dtype != torch.float64 for m in scorer.models for q in m.parameters()):
            raise ValueError('Actual proposal models are not entirely double.')
        for case, time, latent, source_error in operators:
            parent = case['parent_id']
            q = np.zeros(latent.size)
            ad = evaluate(denoiser, scorer, latent, time, q, torch.float64, True)
            gradient = latent.project(ad.pop('gradient'))
            native = evaluate(denoiser, scorer, latent, time, q, torch.float64)
            folder = root/(Path(parent).stem+'-t'+str(int(time*1000)))
            folder.mkdir()
            clean_path = folder/'clean.vasp'
            clean = Atoms(numbers=latent.numbers, cell=np.array(ad['clean_cell'])[0], scaled_positions=ad['clean_frac'], pbc=True)
            write(clean_path, clean, format='vasp', direct=True, sort=False)
            reread = Structure.from_file(clean_path)
            if tuple(reread.atomic_numbers) != tuple(latent.numbers):
                raise ValueError('Saved denoised pose changed species.')
            gap, ef = official.predict(clean_path)
            row = dict(parent_id=parent, time=time, source_pose_sha256=case['B']['relaxed_sha256'],
                source_CF_error=source_error, score=ad, gradient=gradient.tolist(),
                corrected_vs_native_forward=max(abs(ad[k]-native[k]) for k in ('gap', 'eform')),
                same_pose_CF_error=max(abs(gap-ad['gap']), abs(ef-ad['eform'])), official_same_pose_CF=dict(gap=gap, eform=ef),
                clean_pose=str(clean_path), clean_pose_sha256=sha(clean_path),
                latent=dict(numbers=latent.numbers.tolist(), cell=latent.cell.tolist(), frac=latent.frac.tolist()), checks=[])
            index = int(parent.split('-')[1].split('.')[0])
            rng = np.random.default_rng([POLICY['seed'], index, int(time*1000), 77])
            for block in ('atomic', 'strain', 'both'):
                direction = latent.project(rng.normal(size=latent.size))
                if block == 'atomic':
                    direction[-6:] = 0
                elif block == 'strain':
                    direction[:-6] = 0
                direction /= np.linalg.norm(direction)
                predicted = float(gradient@direction)
                for epsilon in DOUBLE_POLICY['FD_steps']:
                    plus = evaluate(denoiser, scorer, latent, time, epsilon*direction, torch.float64)['loss']
                    minus = evaluate(denoiser, scorer, latent, time, -epsilon*direction, torch.float64)['loss']
                    fd = (plus-minus)/(2*epsilon)
                    ok = abs(fd-predicted) <= DOUBLE_POLICY['FD_absolute']+DOUBLE_POLICY['FD_relative']*max(abs(fd), abs(predicted))
                    row['checks'].append(dict(block=block, epsilon=epsilon, plus_loss=plus, minus_loss=minus, FD=fd, predicted=predicted, passed=bool(ok)))
            row['passed'] = case_passes(row)
            result['cases'].append(row)
            save(root/'native.json', result)
            print(json.dumps(dict(parent=parent, time=time, passed=row['passed'], same_pose_CF_error=row['same_pose_CF_error'])), flush=True)
            if not row['passed']:
                raise ValueError('Actual F64 execution native case failed.')
        if len(result['cases']) != 8 or sum(len(q['checks']) for q in result['cases']) != 48:
            raise ValueError('Missing actual F64 native coverage.')
        result['passed'] = True
    except Exception as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save(root/'native.json', result)


if __name__ == '__main__':
    main()
