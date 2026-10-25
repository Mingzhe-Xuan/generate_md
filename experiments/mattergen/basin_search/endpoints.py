"""Package actual audited endpoints and ASE comparisons, without model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from policy import ACCEPTANCE


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_files(first_path,first_sha,second_path,second_sha,folder,labels):
    from ase.io import read,write
    if sha(first_path)!=first_sha or sha(second_path)!=second_sha:raise ValueError('Endpoint source SHA changed.')
    folder.mkdir(parents=True,exist_ok=False)
    poses=[]
    for source,name,label in zip((first_path,second_path),('direct.vasp','selected.vasp'),labels):
        path=folder/name;shutil.copyfile(source,path);atoms=read(path,format='vasp');atoms.info.update(label);poses.append(atoms)
    if not np.array_equal(poses[0].numbers,poses[1].numbers):raise ValueError('Ordered species changed.')
    write(folder/'comparison.traj',poses)
    restored=read(folder/'comparison.traj',index=':')
    if len(restored)!=2:raise ValueError('ASE frame count changed.')
    for a,b in zip(poses,restored):
        np.testing.assert_array_equal(a.numbers,b.numbers)
        np.testing.assert_allclose(a.positions,b.positions,atol=1e-12,rtol=0)
        np.testing.assert_allclose(a.cell.array,b.cell.array,atol=1e-12,rtol=0)
    return poses


def display_pair(poses):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.core import Lattice
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import internal_vectors
    first,second=[a.copy() for a in poses]
    for a in (first,second):
        original=a.copy();cell,_=a.cell.standard_form();a.set_cell(cell,scale_atoms=True)
        np.testing.assert_allclose(original.get_all_distances(mic=True),a.get_all_distances(mic=True),atol=1e-8)
    first_structure=AseAtomsAdaptor.get_structure(first);second_structure=AseAtomsAdaptor.get_structure(second)
    vectors=internal_vectors(first_structure,second_structure)
    # internal_vectors uses pymatgen's canonical lattice orientation, not ASE's.
    canonical=Lattice.from_parameters(*first_structure.lattice.abc,*first_structure.lattice.angles).matrix
    x=first.get_scaled_positions(wrap=False);anchor=int(np.argmax(first.numbers));display=np.remainder(x+.5-x[anchor],1.)
    first.set_scaled_positions(display);second.set_scaled_positions(display+vectors@np.linalg.inv(canonical))
    for a,original in zip((first,second),poses):np.testing.assert_allclose(a.get_all_distances(mic=True),original.get_all_distances(mic=True),atol=1e-8)
    return first,second


def plot_pairs(pairs,output):
    if not pairs:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms
    from ase.data import covalent_radii
    from ase.utils import rotate
    from itertools import product
    fig,axes=plt.subplots(len(pairs),2,figsize=(11,4.3*len(pairs)),squeeze=False,layout='constrained')
    rotation='20x,-25y,10z';q=rotate(rotation)
    for index,(poses,row) in enumerate(pairs):
        first,second=display_pair(poses)
        radii=covalent_radii[first.numbers]*.55;corners=np.array(list(product((0,1),repeat=3)))
        vertices=np.vstack([np.vstack((a.positions,corners@a.cell.array))@q for a in (first,second)])
        lower=vertices[:,:2].min(0)-radii.max()-.5;upper=vertices[:,:2].max(0)+radii.max()+.5
        center=(lower+upper)/2;side=max(upper-lower);bbox=(*tuple(center-side/2),*tuple(center+side/2))
        for col,a in enumerate((first,second)):
            ax=axes[index,col];plot_atoms(a,ax=ax,rotation=rotation,show_unit_cell=2,radii=radii,bbox=bbox)
            gap=row['baseline_prediction' if col==0 else 'prediction']['crystalformer_gap']
            ax.set_title(f'{row["parent_id"]} | '+('Direct MatterSim' if col==0 else 'Beam + MatterSim')+f'\n{a.get_chemical_formula()}, CF gap {gap:.4f} eV, V {a.get_volume():.2f} A^3',fontsize=10)
            ax.set_axis_off()
        axes[index,1].text(.5,-.02,f'Pair-distance RMS {row["geometry"]["species_pair_distance_RMS_A"]:.3f} A; energy delta {1000*row["energy_delta_eV_atom"]:+.1f} meV/atom',
            transform=axes[index,1].transAxes,ha='center',fontsize=9)
    fig.suptitle('All materially changed + CF-improved beam endpoints\nSame projection and scale within each pair; original metric and periodic distances preserved',fontsize=12)
    fig.savefig(output,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Batch endpoint inspection requires Slurm.')
    root=args.run.resolve();load=lambda path:json.loads(path.read_text())
    report=load(root/'report.json');plan=load(root/'plan.json');data=load(root/'search-results.json')
    if report['plan']!=plan or plan['acceptance']!=ACCEPTANCE or len(set(plan['parent_ids']))!=16:raise ValueError('Frozen plan changed.')
    output=root/f'endpoints-{os.environ["SLURM_JOB_ID"]}';output.mkdir(exist_ok=False)
    cases={c['parent_id']:c for c in plan['cases']};manifest=[];beam_pairs=[]
    for method,result in report['methods'].items():
        if [r['parent_id'] for r in result['cases']]!=plan['parent_ids']:raise ValueError('Final cohort missing/reordered.')
        for row,point in zip(result['cases'],data['selected'][method]):
            parent=row['parent_id'];base=cases[parent]['B'];selected=point['B']
            if row['point_id']!=point['id'] or row['prediction']['geometry_sha256']!=selected['relaxed_sha256']:raise ValueError('Final independently scored endpoint differs.')
            labels=[dict(stage=stage,parent_id=parent,method=method,search_seed=report['search_seed'],CF_gap=row[key]['crystalformer_gap'],CF_eform=row[key]['crystalformer_eform'])
                for stage,key in (('direct','baseline_prediction'),('selected','prediction'))]
            relative=Path(method)/Path(parent).stem
            poses=pair_files(base['relaxed_path'],base['relaxed_sha256'],point['pose'],selected['relaxed_sha256'],output/relative,labels)
            manifest.append(dict(method=method,parent_id=parent,folder=str(relative),baseline_sha256=base['relaxed_sha256'],selected_sha256=selected['relaxed_sha256'],
                neutral_mask=cases[parent]['neutral'],baseline_mechanics=base,selected_mechanics=selected,evaluation=row))
            if method=='beam' and row['material_CF_improvement']:beam_pairs.append((poses,row))
    plot_pairs(beam_pairs,output/'beam-material-comparison.png')
    (output/'manifest.json').write_text(json.dumps(dict(search_seed=report['search_seed'],parent_ids=plan['parent_ids'],pairs=manifest,
        note='Original saved post-quench POSCAR bytes and exact two-frame ASE trajectories. Display only standardizes orientation/origin. No new optimization, scoring, or phase/stability certification.'),indent=2,allow_nan=False)+'\n')
    archive=shutil.make_archive(str(output),'gztar',root_dir=output)
    Path(archive+'.sha256').write_text(sha(archive)+'  '+Path(archive).name+'\n')
    print(json.dumps(dict(output=str(output),pairs=len(manifest),ASE_roundtrip_passed=True,material_beam_pairs=len(beam_pairs))),flush=True)


if __name__=='__main__':main()
