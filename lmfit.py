import numpy as np
import pandas as pd
from scipy import linalg
from statsmodels.regression.linear_model import WLS

def lmfit(E, design, weights=None):
    if not isinstance(E, pd.DataFrame) or not isinstance(design, pd.DataFrame):
        raise ValueError("E和design必须为pandas DataFrame")
    if weights is not None and not isinstance(weights, pd.DataFrame):
        raise ValueError("weights必须为pandas DataFrame")

    if not all(E.columns == design.index):
        raise ValueError("E的列名（样本）与design的索引不一致")
    if weights is not None and not all(weights.columns == E.columns):
        raise ValueError("weights的列名（样本）与E的列名不一致")

    E = E.apply(pd.to_numeric, errors='coerce')
    if weights is not None:
        weights = weights.apply(pd.to_numeric, errors='coerce')

    design_matrix = design

    X = design_matrix.values
    y = E.values
    n_genes, n_samples = y.shape
    n_coefs = X.shape[1]

    if weights is None:
        w = np.ones_like(y)
    else:
        w = weights.values
        w[w <= 0] = np.nan

    QR = linalg.qr(X, mode='full')
    rank = np.linalg.matrix_rank(QR[1])
    df_residual = n_samples - rank

    coefficients = np.full((n_genes, n_coefs), np.nan)
    stdev_unscaled = np.full((n_genes, n_coefs), np.nan)
    sigma = np.full(n_genes, np.nan)
    df_resid = np.full(n_genes, df_residual)

    for i in range(n_genes):
        y_i = y[i, :]
        w_i = w[i, :] if weights is not None else np.ones(n_samples)

        valid = np.isfinite(y_i) & np.isfinite(w_i)
        if np.sum(valid) < n_coefs:
            continue

        y_valid = y_i[valid]
        X_valid = X[valid, :]
        w_valid = w_i[valid]

        w_sqrt = np.sqrt(w_valid)
        X_weighted = X_valid * w_sqrt[:, np.newaxis]
        y_weighted = y_valid * w_sqrt

        cond_num = np.linalg.cond(X_weighted)
        if cond_num > 1e12:
            stdev_unscaled[i, :] = np.nan
            continue

        try:
            qr_result = linalg.qr(X_weighted, mode='economic')
            R_weighted = qr_result[1]

            R_inv = linalg.solve_triangular(R_weighted, np.eye(R_weighted.shape[0]), lower=False)
            cov_coef = R_inv @ R_inv.T
            stdev_unscaled[i, :] = np.sqrt(np.diag(cov_coef))
        except:
            stdev_unscaled[i, :] = np.nan

        model = WLS(y_valid, X_valid, weights=w_valid)
        results = model.fit(method='qr')
        coefficients[i, :] = results.params

        if results.df_resid > 0:
            sigma[i] = np.sqrt(results.scale)

    return {
        'coefficients': pd.DataFrame(coefficients, index=E.index, columns=design_matrix.columns),
        'stdev_unscaled': pd.DataFrame(stdev_unscaled, index=E.index, columns=design_matrix.columns),
        'sigma': pd.Series(sigma, index=E.index, name='sigma'),
        'df_residual': pd.Series(df_resid, index=E.index, name='df_residual'),
        'design': design_matrix,
        'rank': rank
    }
