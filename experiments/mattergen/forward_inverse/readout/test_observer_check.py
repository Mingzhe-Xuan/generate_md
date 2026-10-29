import unittest
from unittest.mock import patch
import numpy as np
from extract import checked_embedding


class ObserverCheckTests(unittest.TestCase):
    def test_failed_score_diagnostic_is_not_upgraded_by_feature_success(self):
        first=dict(score=np.array([1.,2.]),graph_hash='same')
        second=dict(score=np.array([2.,1.]),graph_hash='same')
        with patch('extract.hook_embedding',return_value=(np.ones(3),first)),patch('extract.wrapper_embedding',return_value=(np.ones(3),second)):
            result=checked_embedding(None,None,None,None,None)[-1]
        self.assertTrue(result['feature_QC_passed'])
        self.assertFalse(result['original_score_relative_criterion_passed'])
        self.assertGreater(result['score_relative_error'],.1)

    def test_feature_disagreement_and_graph_change_reject(self):
        first=dict(score=np.ones(3),graph_hash='a')
        with patch('extract.hook_embedding',return_value=(np.ones(3),first)),patch('extract.wrapper_embedding',return_value=(np.ones(3)*2,first)):
            with self.assertRaisesRegex(ValueError,'capture'):checked_embedding(None,None,None,None,None)
        with patch('extract.hook_embedding',return_value=(np.ones(3),first)),patch('extract.wrapper_embedding',return_value=(np.ones(3),dict(score=np.ones(3),graph_hash='b'))):
            with self.assertRaisesRegex(ValueError,'capture'):checked_embedding(None,None,None,None,None)


if __name__=='__main__':unittest.main()
