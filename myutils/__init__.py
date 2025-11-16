"""
grstool: A toolkit for genetic risk scoring (GRS) feature selection and model optimization.
"""


import myutils.opitimize
import myutils.draw


evaluate_subset_auc=myutils.opitimize.evaluate_subset_auc
backward_elimination=myutils.opitimize.backward_elimination
GRS_pic=myutils.draw.GRS_pic
ROC_pic=myutils.draw.ROC_pic

__all__ = ['evaluate_subset_auc',  'backward_elimination', 'GRS_pic', 'ROC_pic']