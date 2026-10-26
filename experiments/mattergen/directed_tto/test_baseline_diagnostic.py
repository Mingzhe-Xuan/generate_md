"""Path variability must neither conceal nor fabricate a frozen-pose model mismatch."""
import copy
from baseline_diagnostic import compare


def test_fixed_pose_energy_force_and_path_identity_are_separate_controls():
    case=dict(parent_id='p',B=dict(checkpoint_sha256='pinned',final_energy_eV=-4.,num_atoms=2,max_raw_atomic_force=.005,optimizer_steps=721))
    point=dict(checkpoint_sha256='pinned',energy_eV=-4.,forces_eV_A=[[.005,0,0],[0,0,0]])
    replay=dict(status='converged',final_energy_eV=-4.004,optimizer_steps=124)
    row=compare(case,point,replay,False)
    assert row['fixed_pose_control']['passed'] and not row['raw_path_reproduced']
    altered=copy.deepcopy(point);altered['energy_eV']=-3.
    row=compare(case,altered,dict(replay,final_energy_eV=-4.),True)
    assert not row['fixed_pose_control']['passed'] and row['raw_path_reproduced']
