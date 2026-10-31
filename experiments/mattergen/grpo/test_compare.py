import unittest
from compare import counts_for_test, paired_cluster_interval, composition_summary, audit_arm, checkpoint_for_arm


class ComparisonContracts(unittest.TestCase):
    def test_native_grpo_control_uses_validation_selected_checkpoint(self):
        self.assertEqual(checkpoint_for_arm('grpo_best_native'), 'best.pt')
        self.assertEqual(checkpoint_for_arm('grpo_last'), 'latest.pt')
        self.assertIsNone(checkpoint_for_arm('conditional_native'))
        with self.assertRaises(ValueError): checkpoint_for_arm('unplanned_arm')

    def test_missing_candidates_cannot_silently_shrink_the_benchmark(self):
        events = [dict(index=i, batch=0) for i in range(8)]
        poses = [dict(index=i, numbers=[1]*2) for i in range(8)]
        audit_arm(events, poses, [2]*8)
        with self.assertRaises(ValueError): audit_arm(events[:-1], poses, [2]*8)
        with self.assertRaises(ValueError): audit_arm(events+[events[0]], poses, [2]*8)
        poses[0]['numbers'] = [1]
        with self.assertRaises(ValueError): audit_arm(events, poses, [2]*8)

    def test_composition_diversity_reduces_formula_units_and_keeps_coverage(self):
        rows = [dict(index=0, numbers=[11, 17]), dict(index=1, numbers=[11, 17, 11, 17]),
                dict(index=2, numbers=[101])]
        s = composition_summary(rows, {0, 1})
        self.assertEqual(s['distinct_reduced_compositions'], 1)
        self.assertEqual(s['evaluable'], 2)
        self.assertEqual(s['entropy_nats'], 0.)

    def test_final_counts_draw_only_test_contexts_and_are_reproducible(self):
        rows = [dict(role='train', numbers=[1]*20), dict(role='test', numbers=[1]*4)]
        self.assertEqual(counts_for_test(rows, 8, 42), [4]*8)

    def test_cluster_pairing_and_denominator_preserve_failures(self):
        a = [dict(index=i, batch=i//2, pass_gap=i in (0, 1, 2)) for i in range(4)]
        b = [dict(index=i, batch=i//2, pass_gap=False) for i in range(4)]
        value = paired_cluster_interval(a, b, repeats=100)
        self.assertEqual(value['delta'], .75)
        self.assertEqual(value['clusters'], 2)
        self.assertIsNotNone(value['interval'])
        self.assertIsNone(paired_cluster_interval(b, b)['interval'])
        b[0]['index'] = 100
        with self.assertRaises(ValueError): paired_cluster_interval(a, b)


if __name__ == '__main__': unittest.main()
