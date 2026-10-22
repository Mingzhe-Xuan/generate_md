import unittest
from predict_relaxed import prediction_counts


class PredictionTests(unittest.TestCase):
    def test_unconverged_or_bad_geometry_never_counts_as_joint(self):
        good = dict(relaxation_status='converged', relaxed_gap_target_success=True,
                    relaxed_eform_target_success=True, relaxed_geometry_valid=True)
        unconverged = dict(good, relaxation_status='unconverged')
        bad = dict(good, relaxed_geometry_valid=False)
        self.assertEqual(prediction_counts([good,unconverged,bad]),
                         dict(relaxed_bandgap_count=2,relaxed_eform_count=2,relaxed_joint_count=1))


if __name__ == '__main__':
    unittest.main()
