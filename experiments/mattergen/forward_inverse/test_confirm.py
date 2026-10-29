import unittest
import numpy as np
from confirm_curl import random_plane,projected_adjoints
from relations import centered_basis


class ConfirmationTests(unittest.TestCase):
    def test_orthonormal_random_plane_and_adjoint_orientation(self):
        plane=random_plane(centered_basis(4),np.random.default_rng(123))
        np.testing.assert_allclose(plane.T@plane,np.eye(2),atol=1e-14)
        np.testing.assert_allclose(plane.reshape(4,3,2).sum(0),0,atol=1e-14)
        matrix=np.array([[1.,2.],[-3.,4.]])
        ambient=plane@matrix@plane.T
        gradients=np.array([ambient.T@u for u in plane.T])
        np.testing.assert_allclose(projected_adjoints(plane,gradients),matrix,atol=1e-14)


if __name__=='__main__':unittest.main()
