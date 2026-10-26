"""A separate execution proof cannot pass with missing derivatives or score drift."""
import copy
from double_native import case_passes, DOUBLE_POLICY


def fixture():
    return dict(checks=[dict(block=b, epsilon=e, passed=True) for b in ('atomic', 'strain', 'both') for e in DOUBLE_POLICY['FD_steps']],
        gradient=[1., 2.], source_CF_error=0., same_pose_CF_error=0., corrected_vs_native_forward=0.)


def test_complete_double_proof_requires_each_fd_and_same_pose_official_control():
    good = fixture()
    assert case_passes(good)
    variants = []
    for field in ('same_pose_CF_error', 'source_CF_error', 'corrected_vs_native_forward'):
        bad = copy.deepcopy(good)
        bad[field] = 1.01e-5
        variants.append(bad)
    missing = copy.deepcopy(good)
    missing['checks'].pop()
    variants.append(missing)
    duplicate = copy.deepcopy(good)
    duplicate['checks'][-1] = duplicate['checks'][0]
    variants.append(duplicate)
    failed = copy.deepcopy(good)
    failed['checks'][0]['passed'] = False
    variants.append(failed)
    for bad in variants:
        assert not case_passes(bad)
