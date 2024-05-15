## TODO needs to be completed

import sys
import numpy as np
import pandas as pd
import unittest
import logging

from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
# from scipy import stats

sys.path.insert(0, "../")
from utils.evaluation import ModelEvaluation
import utils.evaluation_utils as eu
import utils.settings as s



# Get logger
logger = s.init_logger('__test_evaluation__')


class TestBinaryModelEvaluation(unittest.TestCase):
    """
    test outputs from classlifcation task
    """

    def test_reverse_probability_scores():
        df = pd.DataFrame({
            "y_pred": [0.0, 0.0, 1.0, 1.0 , 0.0],
            "y_proba": [0.90, 0.64, 0.64, 0.9 , 0.99],
        })
        assert eu.reverse_probability_scores(df)["y_proba"].to_list() == [0.1, 0.36, 0.64, 0.9, 0.01]



class TestModelEvaluation(unittest.TestCase):

    ### Test p-value calculation
    # Test if p-value are derived in the correct way, this means that also t vlaues and standard errors have to be correct.
    # Assert that self-calculated p-value is the same as the p-values derived by stats-package for a simple linear regression

    def setUp(self):
        Xy = pd.DataFrame({"target": [0,1,2,3,4,5,6],
                    "var1": [0,21,22,32,42,52,62],
                    "var2": [10,11,21,31,41,51,61]
            })
        self.me_1 = ModelEvaluation(models_trained_ncv=None, Xy=Xy, target_name="target", 
                                    score_metrics=None, cv=None, kfolds=None, seed=None
                                    )
        self.X_exog = MinMaxScaler().fit_transform(Xy.drop("target", axis=1)) 
        self.newX = np.append(np.ones((len(self.X_exog),1)), self.X_exog, axis=1)
        self.y = Xy["target"]
        self.y_pred = pd.Series([1, 1, 2, 3, 6, 7, 8])

    
    def test_negate_scores_from_sklearn_cross_valdiate(self):
        """
        test if negatiation works correctly for model scores from outer folds of ncv
        """
         # expected output
        reference = {
            'test_MAE': [-0.13224787565117258, 0.000000, +0.1255666668136231],
            'test_R2': [0.01312909501742765, 0.4501174953142224, 0.1746695111706701], 
        }

        ## test negatiation, except for R2 due that it is maximized in sklearn.cross-valdiate()
        self.assertEqual(self.me_1.negate_scores_from_sklearn_cross_valdiate(
            {'test_MAE': [0.13224787565117258, 0.000000, -0.1255666668136231],
             'test_R2': [0.01312909501742765, 0.4501174953142224, 0.1746695111706701], 
             }
        ), reference)

        ## test handling 1D-np.array or list as input values
        self.assertEqual(self.me_1.negate_scores_from_sklearn_cross_valdiate(
            {'test_MAE': np.array([0.13224787565117258, 0.000000, -0.1255666668136231]),
             'test_R2': [0.01312909501742765, 0.4501174953142224, 0.1746695111706701]
             }
        ), reference)


    def test_calc_p_values(self):
        """
        test self calculated p-values of regression coefficients
        """
        coefs_intercept = np.array([7.31132447, -0., 0.77320976 ]) 
        sd_b = self.me_1.calc_standard_error(self.y, self.y_pred, self.newX) 
        ts_b = coefs_intercept / sd_b
        p_values = self.me_1.calc_p_values(ts_b, self.newX)

        self.assertEqual(list(np.round(p_values, 4)), [0.0123, 1.0, 0.9164])


    def test_calc_standard_error(self):
        """
        test calculation of standard error
        """
        sd_1 = self.me_1.calc_standard_error(self.y, self.y_pred, self.newX) 
        self.assertEqual(sd_1.tolist(), [1.6872424731962818, 7.899599466161708, 6.91506481334833])


#     def test_calc_regression_coefficients():
#         """
#         # TODO verify final result from calc_regression_coefficients() with statsmodels-package with a linear regression
#         """
#         X_exog = MinMaxScaler().fit_transform(self.X)#, 
#         y = self.y

#         ## reference: p-values from statsmodels
#         m = sm.OLS(self.y, sm.add_constant(X_exog))
#         m_res = m.fit()
#         #print(m_res.summary())
#         p_values_reference = m_res.summary2().tables[1]['P>|t|']
        
#         # OR with ElasticNEt
#         # from sklearn.linear_model import ElasticNet
#         # from sklearn2pmml.statsmodels import StatsModelsRegressor
#         # from statsmodels.api import OLS
#         # m = StatsModelsRegressor(OLS, fit_intercept = True)
#         # m.fit(X_exog, y, fit_method="fit_regularized", method="elastic_net")
        
#         ## self calculated p-values
#         # reg = ElasticNet().fit(X_exog, self.y)
#         reg = LinearRegression().fit(X_exog, self.y)
#         y_pred_test = reg.predict(X_exog)
#         coefs_intercept = np.append(reg.intercept_, list(reg.coef_))

#         ## calc p-values
#         newX = np.append(np.ones((len(X_exog),1)), X_exog, axis=1)
#         sd_b = self.calc_standard_error(self.y, y_pred_test, newX)  # standard error calculated based on MSE of newX
#         ts_b = coefs_intercept / sd_b        # t values
#         p_v = self.calc_p_values(ts_b, newX)   # significance
#         # print(np.round(p_values_reference, 3), np.round(p_v, 3))

#         assert (list(np.round(p_values_reference, 3)) == np.round(p_v, 3)).all(), logger.critical("wrong calculation of p values!") & sys.exit("stop calculation")
#         print("Passed test for p-value calculation:", (list(np.round(p_values_reference, 3)) == np.round(p_v, 3)).all() )


# # unittest.main(argv=[''], verbosity=2, exit=False)  # when run in cell of jupyter nb

# if __name__ == '__main__':
#     unittest.main() 