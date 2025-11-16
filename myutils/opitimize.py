import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

"""
用于遗传风险评分（GRS）特征选择和模型优化。
For feature selection and model optimization in genetic risk scoring (GRS).
"""


def evaluate_subset_auc(X, y,selected_features='all', cv=5):
    """
    Calculate the auc of selected features from backward elimination.
    
    Parameters
    ----------
    X : Dataframe
        Geno dataframe. Snps in columns and individuals in rows. The values are snp*beta. 
    y:  list
        The state of individual (disease or healthy). With the same order of the rows in X.
    selected_features: str or list
        A list of snps been selected after optimized. 'all':all the snps in X columns.
    cv: int (default=5) 
        Fold number for cross-validation.
    
    Returns
    -------
    list of floats
        The list of auc from all folds
    float
        The mean auc of the optimized model.
    """

    # 异常检查
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.shape[0] != len(y):
        raise ValueError(f"Number of rows in X ({X.shape[0]}) does not match length of y ({len(y)}).")

    if selected_features=='all':
        selected_features= X.columns.tolist()
    elif any(f not in X.columns for f in selected_features):
        missing = [f for f in selected_features if f not in X.columns]
        raise ValueError(f"Find feactures not exist in X: {missing}")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_test = X.iloc[test_idx][selected_features]
        grs_test = X_test.sum(axis=1)
        aucs.append(roc_auc_score(np.array(y)[test_idx], grs_test))
    return np.mean(aucs)



def backward_elimination(X, y,min_features=5, cv=5):
    """
    Calculate the auc of selected features from backward elimination.
    
    Parameters
    ----------
    X : Dataframe
        Geno dataframe. Snps in columns and individuals in rows. The values are snp*beta. 
    y:  list
        The state of individual (disease or healthy). With the same order of the rows in X.
    min_features: int (default=5)
        Minimum number of feature you want to save
    cv: int (default=5)
        Fold number for cross-validation.
    Returns
    -------
    selected: list
        list of selected snps
    history:  list
        The eliminated snps in each iteration and the auc
        

        The picture of distribution curve for GRS.
    """

    selected = list(X.columns)
    best_auc = evaluate_subset_auc(X, y,selected, cv=cv)
    history = [(selected.copy(), best_auc)]

    print(f"Initial AUC with all features ({len(selected)} loci): {best_auc:.4f}")

    while len(selected) > min_features:
        auc_drop = []
        for feat in selected:#依次删除
            trial = [f for f in selected if f != feat]
            auc = evaluate_subset_auc(X, y, trial, cv=cv)#5折计算AUC
            auc_drop.append((feat, auc))

        # 找出删除后 AUC 最大的（说明该位点对模型无贡献或负贡献）
        worst_feature, new_auc = max(auc_drop, key=lambda x: x[1])

        if new_auc > best_auc:#如果优于best_auc,则删除所选位点
            selected.remove(worst_feature)
            best_auc = new_auc
            history.append((selected.copy(), best_auc))
            print(f"Dropped {worst_feature}, AUC improved to {best_auc:.4f}")
        else:
            print("No further AUC improvement by deleting features.")
            break

    return selected, history



if __name__ == '__main__':
    # 构造模拟数据
    np.random.seed(42)
    n_samples = 100
    n_snps = 10
    X = pd.DataFrame(
        np.random.randn(n_samples, n_snps),
        columns=[f'SNP_{i}' for i in range(n_snps)]
    )# 模拟已经乘过beta的值
    
    y = np.random.choice([0, 1], size=n_samples)  # 0: healthy, 1: disease

    # 测试 evaluate_subset_auc
    selected_features = X.columns[:5].tolist()
    auc = evaluate_subset_auc(X, y, selected_features, cv=5)
    print(f'Test AUC for selected features: {auc:.4f}')

    # 测试 backward_elimination
    selected, history = backward_elimination(X, y, min_features=3, cv=5)
    print(f'Backward elimination selected features: {selected}')
    print('Elimination history:')
    for feats, auc_val in history:
        print(f'Features: {feats}, AUC: {auc_val:.4f}')