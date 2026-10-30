"""Audit and paired structure/energy/cost comparison after early CSP stopping."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tarfile
import shutil
import numpy as np
from comparison import classify,eligible,stats,native_force_norm
from geometry import compare,support,effect

ARMS=('mean800','sample800','mean1000')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main(root):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.filters import ExpCellFilter
    from ase.io import read
    from pymatgen.core import Structure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((root/'complete.json').read_text());plan=json.loads((root/'plan.json').read_text())
    if (root/'report.json').exists() and 'endpoint_support' not in json.loads((root/'report.json').read_text()):
        backup=root.with_name(root.name+'-force-only-report-'+sha(root/'report.json')[:12])
        backup.mkdir(exist_ok=True)
        for name in ('report.json','REPORT.md','comparison.png','comparison.svg','evidence.tar.gz','evidence.tar.gz.sha256'):
            if (root/name).exists() and not (backup/name).exists():shutil.copyfile(root/name,backup/name)
    target_fmax=plan['policy']['fmax'];acceptance_fmax=plan['policy'].get('force_acceptance_fmax',.01)
    inputs={(r['trajectory'],r['arm']):r for r in data['inputs']}
    rows={(r['trajectory'],r['arm']):r for r in data['relaxations']}
    expected={(i,a) for i in range(24) for a in ARMS}
    if set(inputs)!=expected or set(rows)!=expected or len(data['inputs'])!=72 or len(data['relaxations'])!=72:
        raise ValueError('Incomplete/duplicated input or relaxation cohort.')
    if (root/'coarse-complete.json').exists():
        if sha(root/'coarse-complete.json')!=plan['source_sha256']['complete.json']:
            raise ValueError('Coarse source hash changed.')
        coarse=json.loads((root/'coarse-complete.json').read_text())
        coarse_rows={(r['trajectory'],r['arm']):r for r in coarse['relaxations']}
        if set(coarse_rows)!=expected:raise ValueError('Refinement omitted a coarse endpoint.')
        for key,row in rows.items():
            if row['coarse_result']!=coarse_rows[key]['result']:raise ValueError('Coarse record changed.')
            with np.load(root/inputs[key]['path']) as d:
                for field in ('numbers','cell','frac'):
                    if not np.array_equal(d[field],row['coarse_result'][field]):raise ValueError('Refinement start differs from coarse endpoint.')
    ledger={};nfe=0
    with (root/'physics/ledger.jsonl').open() as f:
        for line in f:
            entry=json.loads(line)
            for task,result in zip(entry['tasks'],entry['results']):
                if task['id'] in ledger or task['id']!=result['id']:raise ValueError('Duplicate/misaligned ledger.')
                ledger[task['id']]=(task,result);nfe+=result['NFE']
    if nfe!=data['NFE'] or len(ledger)!=data['requests']:raise ValueError('NFE accounting mismatch.')
    structures={};initials={};endpoint_support={};filter_error=0.;output_sha={}
    for key,item in inputs.items():
        path=root/item['path']
        if sha(path)!=item['sha256']:raise ValueError('Input hash changed.')
        with np.load(path) as d:
            numbers=d['numbers'];cell=d['cell'];frac=d['frac']
        initial=Structure(cell,numbers,frac);initials[key]=initial
        row=rows[key];point=row['result']
        if row['force_eligible']!=eligible(point,acceptance_fmax):raise ValueError('Eligibility mismatch.')
        if point['status']=='unknown':
            if point['id'] in ledger or not point['reasons']:raise ValueError('Unknown was evaluated/has no reason.')
            continue
        task,stored=ledger[point['id']]
        if stored!=point or task['action']!='relax' or task['fmax']!=target_fmax or task['max_steps']!=1600 or task['cell_relax'] is not True:
            raise ValueError('Physics settings/result changed.')
        for name,value in (('numbers',numbers),('cell',cell),('frac',frac)):
            if not np.array_equal(task[name],value):raise ValueError('Relaxation used wrong input.')
        folder=Path(task['output']);record_path=folder/'result.json'
        if record_path.exists():
            if json.loads(record_path.read_text())!=point:raise ValueError('Saved relaxation record differs.')
            output_sha[str(record_path.relative_to(root))]=sha(record_path)
        elif point['status']!='failed':
            raise ValueError('Nonfailed endpoint is missing its saved record.')
        if point['status'] not in ('converged','unconverged'):continue
        if point['numbers']!=numbers.tolist() or point['preserve_symmetry'] is not False or point['cell_relax'] is not True:
            raise ValueError('Ordered species/profile changed.')
        atoms=Atoms(numbers=point['numbers'],cell=point['cell'],scaled_positions=point['frac'],pbc=True)
        forces=np.array(point['forces_eV_A']);raw=native_force_norm(point['forces_eV_A'])
        if abs(raw-point['max_raw_force'])>1e-12:raise ValueError('Raw force audit failed.')
        atoms.calc=SinglePointCalculator(atoms,energy=point['energy_eV'],forces=forces,stress=np.array(point['stress_eV_A3']))
        filtered=ExpCellFilter(atoms);filtered.orig_cell=cell.copy()
        maximum=float(np.linalg.norm(filtered.get_forces(),axis=1).max())
        error=abs(maximum-point['max_filter_force']);filter_error=max(filter_error,error)
        if error>1e-9:raise ValueError('Independent filter-force/stress reconstruction failed.')
        vasp=read(folder/'relaxed.vasp',format='vasp')
        if not np.array_equal(vasp.numbers,numbers) or not np.allclose(vasp.cell.array,point['cell'],atol=1e-12,rtol=0):
            raise ValueError('Output VASP cell/species changed.')
        delta=vasp.get_scaled_positions(wrap=False)-np.array(point['frac']);delta-=np.rint(delta)
        if np.max(np.abs(delta))>1e-12:raise ValueError('Output VASP coordinates changed.')
        structures[key]=Structure(point['cell'],point['numbers'],point['frac'])
        endpoint_support[key]=support(point['numbers'],point['cell'],point['frac'])
        output_sha[str(folder.relative_to(root))+'/relaxed.vasp']=sha(folder/'relaxed.vasp')
    if data['native_forwards']!=6000 or any(a['score_calls']!=2000 or a['replacements']!=2000 or not a['rng_unchanged'] or not a['next_corrector_is_sample800'] for a in data['sampling_audit']):
        raise ValueError('Sampler audit failed.')
    branches={};pairs={}
    for arm in ARMS:
        branch=[rows[i,arm] for i in range(24)]
        valid=[r for r in branch if r['force_eligible'] and not endpoint_support[r['trajectory'],arm]['reasons']]
        branches[arm]=dict(total=24,statuses=dict(Counter(r['result']['status'] for r in branch)),force_converged=sum(r['force_eligible'] for r in branch),
            raw_force_converged=sum(r['force_eligible'] for r in branch),valid_endpoint=len(valid),
            geometry_invalid=sum(bool(endpoint_support.get((i,arm),{}).get('reasons')) for i in range(24)),
            steps=stats([r['result']['steps'] for r in valid]),seconds=stats([r['result']['seconds'] for r in valid]),
            NFE=sum(r['result']['NFE'] for r in branch),
            attempted_seconds=sum(r['result']['seconds'] for r in branch),
            initial_max_force=stats([r['result']['initial_max_raw_force'] for r in valid]),
            relaxation_internal_RMS_A=stats([effect.measure(initials[r['trajectory'],arm],structures[r['trajectory'],arm])['internal_RMS_A'] for r in valid]))
        generation_seconds=sum(a['seconds1000' if arm=='mean1000' else 'seconds800'] for a in data['sampling_audit'])
        branches[arm].update(native_generation_seconds=generation_seconds,
            estimated_generation_plus_relaxation_seconds=generation_seconds+branches[arm]['attempted_seconds'])
        if any('coarse_result' in r for r in branch):
            branches[arm].update(cumulative_steps=stats([r['result']['steps']+r['coarse_result']['steps'] for r in valid]),
                cumulative_relaxation_seconds=sum(r['result']['seconds']+r['coarse_result']['seconds'] for r in branch),
                cumulative_NFE=sum(r['result']['NFE']+r['coarse_result']['NFE'] for r in branch))
            branches[arm]['estimated_generation_plus_relaxation_seconds']=generation_seconds+branches[arm]['cumulative_relaxation_seconds']
    for arm in ('mean800','sample800'):
        cases=[]
        for i in range(24):
            early,full=rows[i,arm],rows[i,'mean1000'];case=dict(trajectory=i,system=early['system'],early_status=early['result']['status'],full_status=full['result']['status'])
            if not early['force_eligible'] or not full['force_eligible']:
                case.update(status='not_both_converged');cases.append(case);continue
            if endpoint_support[i,arm]['reasons'] or endpoint_support[i,'mean1000']['reasons']:
                case.update(status='endpoint_geometry_invalid',early_geometry=endpoint_support[i,arm],full_geometry=endpoint_support[i,'mean1000'])
                cases.append(case);continue
            n=len(structures[i,arm]);delta=(early['result']['energy_eV']-full['result']['energy_eV'])/n
            geom=compare(structures[i,'mean1000'],structures[i,arm])
            case.update(status='compared',**classify(delta,geom['strict_match'],geom['conventional_match']),geometry=geom,
                steps_early=early['result']['steps'],steps_full=full['result']['steps'],
                seconds_early=early['result']['seconds'],seconds_full=full['result']['seconds'],
                energy_early_eV_atom=early['result']['energy_eV']/n,energy_full_eV_atom=full['result']['energy_eV']/n)
            cases.append(case)
        valid=[c for c in cases if c['status']=='compared']
        pairs[arm]=dict(cases=cases,both_converged=len(valid),strict_match=sum(c['strict_match'] for c in valid),
            conventional_match=sum(c['conventional_match'] for c in valid),consistent_endpoint=sum(c['consistent_endpoint'] for c in valid),
            energy_winners=dict(Counter(c['energy_winner'] for c in valid)),
            delta_energy_eV_atom=stats([c['delta_energy_eV_atom'] for c in valid]),
            absolute_delta_energy_eV_atom=stats([abs(c['delta_energy_eV_atom']) for c in valid]),
            delta_volume_percent=stats([c['geometry']['volume_change_percent'] for c in valid]),
            ordered_internal_RMS_A=stats([c['geometry']['ordered']['internal_RMS_A'] for c in valid]),
            aligned_internal_RMS_A=stats([c['geometry']['aligned_internal_RMS_A'] for c in valid if c['geometry']['aligned_internal_RMS_A'] is not None]))
    report=dict(audit=dict(passed=True,inputs=72,requests=len(ledger),NFE=nfe,max_filter_force_reconstruction_error=filter_error),
        branches=branches,pairs=pairs,sampling_audit=data['sampling_audit'],policy=plan['policy'],
        endpoint_support={f'{i}:{arm}':value for (i,arm),value in endpoint_support.items()},
        input_sha256={n:sha(root/n) for n in ('plan.json','complete.json','physics/ledger.jsonl')},output_sha256=output_sha,
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},elapsed_seconds=data['elapsed_seconds'])
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    colors={'mean800':'#1769a4','sample800':'#cf661d'}
    for arm in ('mean800','sample800'):
        valid=[c for c in pairs[arm]['cases'] if c['status']=='compared'];color=colors[arm]
        axes[0,0].scatter([c['energy_full_eV_atom'] for c in valid],[c['energy_early_eV_atom'] for c in valid],label=arm,alpha=.75,color=color)
        axes[0,1].hist([1000*c['delta_energy_eV_atom'] for c in valid],bins=12,label=arm,alpha=.45,color=color)
        axes[1,0].scatter([c['steps_full'] for c in valid],[c['steps_early'] for c in valid],label=arm,alpha=.75,color=color)
        axes[1,1].scatter([1000*c['delta_energy_eV_atom'] for c in valid],[c['geometry']['pair_distance_signature_RMS_A'] for c in valid],label=arm,alpha=.75,color=color)
    for ax in (axes[0,0],axes[1,0]):
        lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
    axes[0,0].set(title='Final MatterSim energies',xlabel='1000-step mean + relax (eV/atom)',ylabel='800-step cutoff + relax (eV/atom)')
    axes[0,1].set(title='Paired energy difference',xlabel='E(800 + relax) - E(1000 + relax), meV/atom',ylabel='Number of pairs');axes[0,1].axvline(0,color='k',lw=.8)
    axes[1,0].set(title='Relaxation effort',xlabel='1000-step relaxation BFGS steps',ylabel='800-step relaxation BFGS steps')
    axes[1,1].set(title='Geometry change versus energy difference',xlabel='Paired energy difference (meV/atom)',ylabel='Species-pair distance signature RMS (A)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('GSS12 / 24 paired CSP paths: exact 800-round cutoff versus native1000 output')
    for suffix in ('png','svg'):fig.savefig(root/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# Exact 800-round CSP cutoff versus full1000, after MatterSim relaxation','',
        'All24GSSparents retained. Main comparison: noise-free-last-predictor mean800 versus native mean1000; '
        'sample800 is the actual stochastic continuation state. The original1000-step time grid is unchanged. '
        'All endpoints come from the same replay. Native CUDA replay differences versus the old run are recorded separately.','',
        f'MatterSim5M, unconstrained coordinates+cell, BFGS/ExpCellFilter, current-stage fmax{target_fmax}, maxstep0.05, cap1600. '
        f'Physical eligibility requires both atomic and filter forces <={acceptance_fmax}. Strict match: ltol.02, stol.05, angle1deg, no rescaling/primitive reduction. '
        'Endpoint consistency additionally requires energy difference <=1meV/atom. This is a numerical criterion, not proof of basin identity.','',
        'The same geometry domain used for trajectory inputs is checked on relaxed endpoints: volume/atom >=2 A^3, '
        'pair distance >=0.5 A, reduced-cell minimum singular value >=0.3 A. Cell reduction avoids mistaking an equivalent skew basis for a singular physical lattice. '
        'Force convergence and geometry support are separate; invalid endpoints remain in all24-parent denominators and in total cost, but not in normal energy/structure comparisons.','',
        '| Branch | Raw/filter force-converged /24 | Force + geometry-valid /24 | Mean current-stage BFGS steps (valid) | Median steps (valid) | Mean seconds (valid) | Current-stage NFE (all) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        b=branches[arm];lines.append(f'| {arm} | {b["raw_force_converged"]}/24 | {b["valid_endpoint"]}/24 | {b["steps"].get("mean")} | {b["steps"].get("median")} | {b["seconds"].get("mean")} | {b["NFE"]} |')
    if any('cumulative_steps' in b for b in branches.values()):
        lines+=['','All72 coarse endpoints received the same tighter continuation, including those already passing the physical force criterion. '
            'The coarse run is preserved separately. The following costs include both relaxation stages; generation cost uses the original three replay batches.','',
            '| Branch | Mean cumulative BFGS steps | Median cumulative BFGS steps | Total cumulative relaxation seconds | Estimated generation + relaxation seconds | Cumulative NFE |',
            '|---|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            b=branches[arm];lines.append(f'| {arm} | {b["cumulative_steps"].get("mean")} | {b["cumulative_steps"].get("median")} | {b["cumulative_relaxation_seconds"]} | {b["estimated_generation_plus_relaxation_seconds"]} | {b["cumulative_NFE"]} |')
    lines+=['','| Early branch vs mean1000 | Both force + geometry-valid | Strict matches | Conventional matches | Strict + <=1meV/atom | Mean signed delta (meV/atom) | Median absolute delta (meV/atom) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('mean800','sample800'):
        p=pairs[arm];lines.append(f'| {arm} | {p["both_converged"]}/24 | {p["strict_match"]} | {p["conventional_match"]} | {p["consistent_endpoint"]} | {1000*p["delta_energy_eV_atom"].get("mean",float("nan"))} | {1000*p["absolute_delta_energy_eV_atom"].get("median",float("nan"))} |')
    lines+=['','| Parent | System | Early branch | Status | Delta E (meV/atom) | Strict match | Aligned internal RMS (A; matched only) | Volume delta % |',
        '|---|---|---|---|---:|---|---:|---:|']
    for arm in ('mean800','sample800'):
        for c in pairs[arm]['cases']:
            g=c.get('geometry',{});e=1000*c['delta_energy_eV_atom'] if 'delta_energy_eV_atom' in c else None
            lines.append(f'| {c["trajectory"]} | {c["system"]} | {arm} | {c["status"]} | {e} | {c.get("strict_match")} | {g.get("aligned_internal_RMS_A")} | {g.get("volume_change_percent")} |')
    lines+=['','Ordered-site internal RMS removes common translation and affine cell deformation but does not permit atom permutation. '
        'Aligned internal RMS permits the strict matcher site mapping and is only available for strict matches. '
        'Sorted species-pair distances are permutation-invariant descriptors, not unique structure identifiers. '
        'Energies are MatterSim predictions, not DFT or convex-hull stability. Paths are coupled within three native batches; descriptive paired statistics only.','',
        f'Raw input/result/VASP/force-stress/NFE audit passed; maximum reconstructed filter-force error {filter_error:.3g}.','',
        '![Comparison](comparison.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in ('evidence.tar.gz','evidence.tar.gz.sha256')]
    with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
        for path in files:archive.add(path,arcname=str(path.relative_to(root)))
    (root/'evidence.tar.gz.sha256').write_text(sha(root/'evidence.tar.gz')+'  evidence.tar.gz\n')
    print(json.dumps(dict(audit=report['audit'],branches=branches,pairs={k:{f:v for f,v in p.items() if f!='cases'} for k,p in pairs.items()},archive_sha256=sha(root/'evidence.tar.gz'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(args.run.resolve())
