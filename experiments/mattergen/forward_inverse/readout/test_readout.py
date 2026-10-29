import unittest
import numpy as np
from regression import fit_head,predict,shuffle_parents,paired_mae_interval
from prepare import ordered_rows


class ReadoutTests(unittest.TestCase):
    def test_svd_ridge_matches_closed_form_and_train_only_scaler(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(30,5));y=rng.normal(size=(30,2))
        vx=rng.normal(size=(10,5))+20;vy=rng.normal(size=(10,2))
        head=fit_head(x,y,vx,vy,alphas=np.array([.3]))
        sx=(x-x.mean(0))/x.std(0)
        exact=np.linalg.solve(sx.T@sx+.3*np.eye(5),sx.T@(y-y.mean(0)))
        np.testing.assert_allclose(head['coefficient'],exact,atol=1e-13)
        np.testing.assert_allclose(head['mean'],x.mean(0),atol=0)
        np.testing.assert_allclose(head['ymean'],y.mean(0),atol=0)

    def test_selection_does_not_use_labels_or_input_row_order(self):
        rows=[dict(material_id='mp-'+str(i),band_gap=i) for i in range(30)]
        first=[r['material_id'] for _,r in ordered_rows(rows,'test')]
        changed=[dict(r,band_gap=1000-r['band_gap']) for r in rows[::-1]]
        second=[r['material_id'] for _,r in ordered_rows(changed,'test')]
        self.assertEqual(first,second)

    def test_parent_shuffle_and_paired_parent_bootstrap(self):
        labels=np.arange(12).reshape(6,2)
        shuffled,p=shuffle_parents(labels)
        np.testing.assert_array_equal(shuffled,labels[p])
        first=np.full((6,4,2),.2);second=np.full((6,4,2),.5);target=np.zeros((6,2))
        result=paired_mae_interval(first,second,target)
        np.testing.assert_allclose(result['mean_MAE_difference'],[-.3,-.3],atol=1e-15)
        np.testing.assert_allclose(result['low95'],[-.3,-.3],atol=1e-15)
        self.assertEqual(result['parents'],6)

    def test_zero_variance_features_and_nonnegative_gap(self):
        x=np.zeros((10,3));y=np.repeat([[-2.,-1.]],10,axis=0)
        head=fit_head(x,y,x,y,alphas=np.array([1.]))
        np.testing.assert_allclose(predict(head,x),np.repeat([[0.,-1.]],10,axis=0))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):fit_head(x*np.nan,y,x,y)


if __name__=='__main__':unittest.main()
