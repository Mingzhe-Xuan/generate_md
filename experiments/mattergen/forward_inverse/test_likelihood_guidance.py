import unittest
import numpy as np
from relations import gaussian_linear_likelihood_guidance


class LikelihoodGuidanceTests(unittest.TestCase):
    def test_posterior_mean_and_variance_match_exact_conditional_score(self):
        clean=np.array([[.7,.2],[.2,.3]])
        noise=np.array([[.3,-.1],[-.1,.5]])
        a=np.array([1.,2.]);x=np.array([.4,-.2]);target=1.3;tau2=.05
        v=np.linalg.inv(np.linalg.inv(clean)+np.linalg.inv(noise))
        dm=clean@np.linalg.inv(clean+noise);mean=dm@x
        actual=gaussian_linear_likelihood_guidance(mean,v,dm,a,target,tau2)
        conditioned_cov=np.linalg.inv(np.linalg.inv(clean)+np.outer(a,a)/tau2)
        conditioned_mean=conditioned_cov@a*target/tau2
        score_cond=-np.linalg.solve(conditioned_cov+noise,x-conditioned_mean)
        score_uncond=-np.linalg.solve(clean+noise,x)
        np.testing.assert_allclose(actual,score_cond-score_uncond,atol=1e-12)
        naive=dm.T@a*(target-a@mean)/tau2
        self.assertGreater(np.linalg.norm(naive),5*np.linalg.norm(actual))


if __name__=='__main__':unittest.main()
