"""Optional observer: serialize current geometry without mutating optimizer atoms."""
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import write


class TrajectoryCapture:
    def __init__(self,folder):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True,exist_ok=False)
        self.path=self.folder/'frames.jsonl'
        self.count=0

    def record(self,atoms,stage,step):
        frame=self.folder/f'frame-{self.count:04d}.vasp'
        energy=float(atoms.get_potential_energy())
        forces=atoms.get_forces(apply_constraint=False)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError('Nonfinite observed trajectory state.')
        copy=atoms.copy();copy.set_constraint(None)
        write(frame,copy,format='vasp',direct=True,sort=False)
        row=dict(frame_index=self.count,stage=stage,B_step=step,path=str(frame.resolve()),
            sha256=hashlib.sha256(frame.read_bytes()).hexdigest(),energy_eV=energy,
            max_raw_force=float(np.linalg.norm(forces,axis=1).max()),num_atoms=len(atoms))
        with self.path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.count+=1
