import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'smoacs/alternating'))
from trajectory_capture import TrajectoryCapture
spec=importlib.util.spec_from_file_location('relaxation_path_report_tests',Path(__file__).with_name('report.py'))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
fixed_mae,monotonicity=report.fixed_mae,report.monotonicity


class PathTests(unittest.TestCase):
    def test_observer_does_not_modify_atoms_constraints_or_cached_result(self):
        atoms=Atoms('Cu2',positions=[[0,0,0],[2,0,0]],cell=[6,6,6],pbc=True)
        atoms.calc=SinglePointCalculator(atoms,energy=-2.,forces=np.array([[.1,0,0],[-.1,0,0]]))
        atoms.set_constraint(FixAtoms(indices=[0]));positions=atoms.positions.copy()
        with tempfile.TemporaryDirectory() as directory:
            capture=TrajectoryCapture(Path(directory)/'trajectory')
            capture.record(atoms,'input',None);capture.record(atoms,'optimizer',0)
            rows=[json.loads(x) for x in capture.path.read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertIsNone(rows[0]['B_step']);self.assertEqual(rows[1]['B_step'],0)
            self.assertEqual(rows[0]['energy_eV'],-2.);self.assertAlmostEqual(rows[0]['max_raw_force'],.1)
            np.testing.assert_array_equal(atoms.positions,positions);self.assertEqual(len(atoms.constraints),1)
            np.testing.assert_array_equal(read(rows[0]['path']).numbers,atoms.numbers)
            self.assertFalse(read(rows[0]['path']).constraints)

    def test_fixed_denominator_carries_finished_trajectory(self):
        a=[dict(stage='optimizer',B_step=i,gap_absolute_error=v) for i,v in enumerate([0.,1.])]
        b=[dict(stage='optimizer',B_step=i,gap_absolute_error=v) for i,v in enumerate([0.,2.,3.])]
        self.assertEqual(fixed_mae([a,b],2),2.)
        self.assertEqual(fixed_mae([a,b],10),2.)

    def test_nonmonotone_error_is_not_reported_as_gradual_increase(self):
        row=monotonicity([.1,.5,.3,.8,.8]);self.assertFalse(row['nondecreasing'])
        self.assertEqual(row['increases'],2);self.assertEqual(row['decreases'],1);self.assertEqual(row['unchanged'],1)


if __name__=='__main__':unittest.main()
