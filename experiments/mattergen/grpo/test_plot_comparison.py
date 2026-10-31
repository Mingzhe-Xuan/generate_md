import copy
import unittest
from compare import ARMS
from plot_comparison import ARM_LABELS, validate_comparison


def fixture():
    metrics = dict(n=8, valid=6, gap_pass=2, valid_fraction=.75, gap_pass_fraction=.25,
                   gap_MAE_valid_eV=.4)
    row = dict(raw=metrics, relaxed=copy.deepcopy(metrics), force_converged=5, joint_gap_force_pass=1)
    return dict(passed=True, rows={arm: copy.deepcopy(row) for arm in ARMS})


class PlotContracts(unittest.TestCase):
    def test_prespecified_arms_full_denominator_and_undefined_mae(self):
        self.assertEqual(tuple(ARM_LABELS), ARMS)
        data = fixture()
        self.assertEqual(validate_comparison(data), 8)
        for stage in ('raw', 'relaxed'):
            data['rows']['base'][stage].update(valid=0, gap_pass=0, valid_fraction=0.,
                                              gap_pass_fraction=0., gap_MAE_valid_eV=None)
        data['rows']['base'].update(joint_gap_force_pass=0)
        self.assertEqual(validate_comparison(data), 8)
        data['rows']['base']['raw']['gap_MAE_valid_eV'] = 0.
        with self.assertRaises(ValueError): validate_comparison(data)

    def test_incomplete_inconsistent_or_nonfinite_reports_reject(self):
        for mode in ('missing_arm', 'unfinished', 'denominator', 'fraction', 'joint', 'nonfinite'):
            data = fixture(); row = data['rows']['base']
            if mode == 'missing_arm': del data['rows']['base_native']
            elif mode == 'unfinished': data['passed'] = False
            elif mode == 'denominator': row['relaxed']['n'] = 6
            elif mode == 'fraction': row['raw']['gap_pass_fraction'] = 2/6
            elif mode == 'joint': row['joint_gap_force_pass'] = 3
            elif mode == 'nonfinite': row['raw']['gap_MAE_valid_eV'] = float('nan')
            with self.subTest(mode=mode), self.assertRaises(ValueError): validate_comparison(data)


if __name__ == '__main__': unittest.main()
