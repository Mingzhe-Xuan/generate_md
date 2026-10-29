import unittest
import numpy as np
from response_report import curl_evidence


class CurlEvidenceTests(unittest.TestCase):
    def test_cross_curl_requires_graph_and_numerical_checks(self):
        j=np.array([[1.,2.],[-3.,4.]])
        v,w=np.eye(2)
        a=np.stack([j@v,j@v]);b=np.stack([j@w,j@w])
        result=curl_evidence(j,v,w,a,b,[True,True])
        self.assertTrue(result['corroborated_nonzero_curl'])
        self.assertFalse(curl_evidence(j,v,w,a,b,[True,False])['corroborated_nonzero_curl'])
        noisy=a+10
        self.assertFalse(curl_evidence(j,v,w,noisy,b,[True,True])['corroborated_nonzero_curl'])

    def test_symmetric_jacobian_has_no_curl_evidence(self):
        j=np.array([[1.,2.],[2.,4.]])
        v,w=np.eye(2)
        a=np.stack([j@v,j@v]);b=np.stack([j@w,j@w])
        result=curl_evidence(j,v,w,a,b,[True,True])
        self.assertTrue(result['numerical_direction_QC'])
        self.assertFalse(result['corroborated_nonzero_curl'])


if __name__=='__main__':unittest.main()
