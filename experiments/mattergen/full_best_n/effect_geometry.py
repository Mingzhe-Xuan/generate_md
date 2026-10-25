"""Physical saved-pose changes, separated from matching tolerances."""
import numpy as np
from pymatgen.core import Lattice
from pymatgen.util.coord import pbc_shortest_vectors


def internal_vectors(first,second):
    if list(first.atomic_numbers)!=list(second.atomic_numbers):raise ValueError('Fixed ordered atoms differ.')
    lattice=Lattice.from_parameters(*first.lattice.abc,*first.lattice.angles)
    x,y=first.frac_coords,second.frac_coords;n=len(x);inverse=np.linalg.inv(lattice.matrix);answers=[]
    for initial in np.vstack([np.zeros(3),x-y]):
        shift=initial.copy()
        for _ in range(40):
            vectors=pbc_shortest_vectors(lattice,x,y+shift)[np.arange(n),np.arange(n)]
            change=vectors.mean(0)@inverse;shift-=change
            if np.linalg.norm(change)<1e-12:break
        vectors=pbc_shortest_vectors(lattice,x,y+shift)[np.arange(n),np.arange(n)]
        answers.append((np.mean(np.sum(vectors*vectors,axis=1)),vectors))
    return min(answers,key=lambda r:r[0])[1]


def distance_signature(structure):
    groups={};matrix=structure.distance_matrix
    for i in range(len(structure)):
        for j in range(i+1,len(structure)):
            key=tuple(sorted((structure[i].specie.symbol,structure[j].specie.symbol)))
            groups.setdefault(key,[]).append(matrix[i,j])
    return np.array([v for key in sorted(groups) for v in sorted(groups[key])])


def measure(first,second):
    distances=np.linalg.norm(internal_vectors(first,second),axis=1)
    signature=distance_signature(second)-distance_signature(first)
    return dict(internal_RMS_A=float(np.sqrt(np.mean(distances**2))),internal_max_A=float(distances.max()),
        species_pair_distance_RMS_A=float(np.sqrt(np.mean(signature**2))),
        volume_change_percent=float(100*(second.volume/first.volume-1)),
        length_change_percent=(100*(np.array(second.lattice.abc)/first.lattice.abc-1)).tolist(),
        angle_change_deg=(np.array(second.lattice.angles)-first.lattice.angles).tolist())
