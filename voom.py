import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import interpolate
import patsy


def choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1 / 3):
    if n_genes <= small_n:
        return 1.0
    else:
        return max(min_span, (small_n / n_genes) ** power)


def normalize_between_arrays(y, method="none"):
    if method == "none":
        return y
    elif method == "quantile":
        ref_dist = np.median(y, axis=0)
        sorted_idx = np.argsort(y, axis=0)
        unsorted_idx = np.argsort(sorted_idx, axis=0)
        return ref_dist[sorted_idx][unsorted_idx]
    elif method == "cyclicloess":
        n_samples = y.shape[1]
        normalized_y = y.copy()
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                ratio = normalized_y[:, i] / normalized_y[:, j]
                x = np.arange(len(ratio))
                mask = ~np.isnan(ratio) & ~np.isinf(ratio)
                if np.sum(mask) < 2:
                    continue
                model = sm.OLS(ratio[mask], sm.add_constant(x[mask]))
                results = model.fit()
                normalized_y[:, i] = normalized_y[:, i] / np.exp(results.fittedvalues)
        return normalized_y
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def voom(counts, design=None, lib_size=None, norm_factors=None,
         normalize_method="none", span=0.5, adaptive_span=False):
    out = {}

    if isinstance(counts, pd.DataFrame):
        counts_df = counts
        counts = counts.values
    else:
        counts = np.asarray(counts)
        counts_df = pd.DataFrame(counts)

    n_genes, n_samples = counts.shape
    if n_genes < 2:
        raise ValueError("需要至少2个基因用于拟合均值-方差趋势")
    if np.isnan(counts).any():
        raise ValueError("计数矩阵中存在NA值")
    if (counts < 0).any():
        raise ValueError("计数矩阵中存在负值")

    if design is None:
        design = np.ones((n_samples, 1))
        design = pd.DataFrame(design, columns=['Intercept'], index=counts_df.columns)
    else:
        if not isinstance(design, pd.DataFrame):
            design = pd.DataFrame(design)
        design = design.reindex(counts_df.columns)

        design = design.apply(pd.to_numeric, errors='coerce')
        if design.isna().any().any():
            raise ValueError("设计矩阵包含非数值或无效值")

    design_arr = design.values.astype(float)

    if lib_size is None:
        lib_size = counts.sum(axis=0)
    else:
        lib_size = np.asarray(lib_size)
    if norm_factors is not None:
        norm_factors = np.asarray(norm_factors)
        if len(norm_factors) != n_samples:
            raise ValueError("归一化因子长度与样本数不匹配")
        lib_size = lib_size * norm_factors

    if adaptive_span:
        span = choose_lowess_span(n_genes)

    y = np.log2((counts + 0.5) / (lib_size + 1) * 1e6).T
    y = normalize_between_arrays(y, method=normalize_method).T

    fit = {'Amean': np.zeros(n_genes), 'sigma': np.zeros(n_genes), 'df.residual': np.zeros(n_genes)}
    for i in range(n_genes):
        gene_expr = y[i, :]
        valid_idx = ~np.isnan(gene_expr)
        if np.sum(valid_idx) < design_arr.shape[1]:
            fit['Amean'][i] = np.nan
            fit['sigma'][i] = np.nan
            fit['df.residual'][i] = 0
            continue
        model = sm.OLS(gene_expr[valid_idx], design_arr[valid_idx, :])
        results = model.fit()
        fit['Amean'][i] = gene_expr[valid_idx].mean()
        fit['sigma'][i] = np.sqrt(results.mse_resid)
        fit['df.residual'][i] = results.df_resid

    n_with_reps = np.sum(fit['df.residual'] > 0)
    if n_with_reps < 2:
        out['E'] = pd.DataFrame(y, index=counts_df.index, columns=counts_df.columns)
        out['weights'] = pd.DataFrame(np.ones_like(y), index=counts_df.index, columns=counts_df.columns)
        out['design'] = design
        return out

    sx = fit['Amean'] + np.mean(np.log2(lib_size + 1)) - np.log2(1e6)
    sy = np.sqrt(fit['sigma'])
    all_zero = (counts.sum(axis=1) == 0)
    sx = sx[~all_zero & ~np.isnan(sx) & ~np.isnan(sy)]
    sy = sy[~all_zero & ~np.isnan(sx) & ~np.isnan(sy)]
    if len(sx) < 2:
        raise ValueError("有效基因数不足，无法拟合均值-方差趋势")
    l = lowess(sy, sx, frac=span, it=3)

    f = interpolate.interp1d(
        l[:, 0], l[:, 1],
        bounds_error=False,
        fill_value=(l[0, 1], l[-1, 1]),
        assume_sorted=True
    )

    fitted_values = np.zeros_like(y)
    for i in range(n_genes):
        if not np.isnan(fit['Amean'][i]):
            gene_expr = y[i, :]
            valid_idx = ~np.isnan(gene_expr)
            if np.sum(valid_idx) >= design_arr.shape[1]:
                model = sm.OLS(gene_expr[valid_idx], design_arr[valid_idx, :])
                results = model.fit()
                fitted_values[i, valid_idx] = results.fittedvalues
                fitted_values[i, ~valid_idx] = np.nan

    fitted_cpm = 2 ** fitted_values
    lib_size_2d = (lib_size + 1).reshape(-1, 1)
    fitted_count = 1e-6 * (fitted_cpm.T * lib_size_2d).T
    fitted_logcount = np.log2(fitted_count)
    w = 1 / (f(fitted_logcount) ** 4)
    w = np.clip(w, a_min=0, a_max=None)

    out['E'] = pd.DataFrame(y, index=counts_df.index, columns=counts_df.columns)
    out['weights'] = pd.DataFrame(w, index=counts_df.index, columns=counts_df.columns)
    out['design'] = design
    return out
