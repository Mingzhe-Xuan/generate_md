import unittest
from measurements import crystalformer_steps, mattersim_steps, describe


class TimingTests(unittest.TestCase):
    def test_progress_elapsed_difference_not_smoothed_rate(self):
        text='step: 0%| | 0/2 [00:00<?, ?it/s]\rstep: 50%| | 1/2 [00:12<00:12, 12.54s/it]\rstep:100%| | 2/2 [00:25<00:00, 12.50s/it]'
        rows=crystalformer_steps(text,2)
        self.assertEqual([r['seconds'] for r in rows],[12,13])
        self.assertEqual(crystalformer_steps(text+'\r'+text.split('\r')[-1],2),rows)
        with self.assertRaises(ValueError):
            crystalformer_steps(text.replace('1/2','0/2'),2)
        with self.assertRaises(ValueError):
            crystalformer_steps(text,3)

    def test_initial_state_midnight_and_second_quantization(self):
        text='BFGS: 0 23:59:59 -1 1\nBFGS: 1 00:00:00 -2 1\nBFGS: 2 00:00:00 -3 0'
        record=mattersim_steps(text,2)
        self.assertEqual(record['logged_updates'],2)
        self.assertEqual(record['mean_seconds_per_step'],.5)
        self.assertEqual([r['seconds'] for r in record['intervals']],[1,0])
        with self.assertRaises(ValueError):
            mattersim_steps(text,3)
        with self.assertRaises(ValueError):
            mattersim_steps(text.replace('BFGS: 2','BFGS: 3'))
        with self.assertRaises(ValueError):
            mattersim_steps('BFGS: 0 12:00:01 -1 1\nBFGS: 1 12:00:00 -2 0')

    def test_empty_zero_and_failed_partial_records(self):
        self.assertIsNone(mattersim_steps('')['logged_updates'])
        with self.assertRaises(ValueError):
            mattersim_steps('',0)
        self.assertIsNone(mattersim_steps('BFGS: 0 12:00:00 -1 0',0)['mean_seconds_per_step'])
        self.assertEqual(mattersim_steps('BFGS: 0 12:00:00 -1 1\nBFGS: 1 12:00:01 -2 1')['logged_updates'],1)

    def test_linear_quantiles_and_constant_counts(self):
        stats=describe([1,2,3,4])
        self.assertEqual(stats['p25'],1.75)
        self.assertEqual(stats['median'],2.5)
        self.assertAlmostEqual(stats['p95'],3.85,places=12)
        self.assertEqual(describe([200]*128)['mean'],200)
        self.assertEqual(describe([])['n'],0)


if __name__ == '__main__':
    unittest.main()
