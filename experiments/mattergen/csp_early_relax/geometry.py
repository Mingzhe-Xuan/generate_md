"""Separate matching tolerances, site displacements and cell changes."""
from collections import Counter
import importlib.util
from pathlib import Path
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher

path=Path(__file__).resolve().parents[1]/'full_best_n/effect_geometry.py'
spec=importlib.util.spec_from_file_location('early_effect_geometry',path)
effect=importlib.util.module_from_spec(spec);spec.loader.exec_module(effect)


def support(numbers,cell,frac):
    from pymatgen.core import Structure
    cell=np.asarray(cell);frac=np.asarray(frac)
    if not np.isfinite(cell).all() or not np.isfinite(frac).all() or np.linalg.det(cell)<=1e-6:
        return dict(reasons=['invalid_cell_or_coordinates'])
    volume=float(np.linalg.det(cell)/len(numbers))
    result=dict(volume_per_atom_A3=volume,reasons=[])
    if volume<2.:
        result['reasons'].append('density_guard_volume_per_atom_below_2A3');return result
    structure=Structure(cell,numbers,frac).get_reduced_structure()
    minimum=float(np.linalg.svd(structure.lattice.matrix,compute_uv=False).min())
    result['reduced_cell_min_singular_A']=minimum
    if minimum<.3:result['reasons'].append('nearly_singular_reduced_cell')
    matrix=structure.distance_matrix;np.fill_diagonal(matrix,np.inf)
    result['minimum_pair_A']=float(matrix.min())
    if result['minimum_pair_A']<.5:result['reasons'].append('collision_below_0.5A')
    return result


def compare(first,second):
    if Counter(first.atomic_numbers)!=Counter(second.atomic_numbers):raise ValueError('Compositions differ.')
    strict=StructureMatcher(ltol=.02,stol=.05,angle_tol=1,primitive_cell=False,scale=False,attempt_supercell=False)
    broad=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    matched=bool(strict.fit(first,second));aligned=None
    if matched:
        mapped=strict.get_s2_like_s1(first,second)
        if mapped is None:raise ValueError('Strict fit cannot be aligned.')
        aligned=effect.measure(first,mapped)['internal_RMS_A']
    ordered=effect.measure(first,second) if list(first.atomic_numbers)==list(second.atomic_numbers) else None
    delta=effect.distance_signature(second)-effect.distance_signature(first)
    return dict(strict_match=matched,conventional_match=bool(broad.fit(first,second)),
        aligned_internal_RMS_A=aligned,ordered=ordered,
        pair_distance_signature_RMS_A=float(np.sqrt(np.mean(delta**2))),
        volume_change_percent=float(100*(second.volume/first.volume-1)))
