import numpy as np
import pandas as pd
from scipy.stats import f
from scipy.stats import t
from limma.core.test_functions import classify_tests_f
from limma.core.squeezeVar import squeezeVar

def eBayes(fit,proportion=0.01,stdev_coef_lim=(0.1,4),trend=False,robust=False,winsor_tail_p=(0.05,0.1),legacy=None):

    if not isinstance(fit, dict):  # 或者换成你自定义的 MArrayLM 类
        raise TypeError("fit is not a valid MArrayLM object")

    if trend is True and fit.get("Amean") is None:
        raise ValueError("Need Amean component in fit to estimate trend")

    # 假设 _ebayes 返回一个字典 eb
    eb = _ebayes(
        fit=fit,
        proportion=proportion,
        stdev_coef_lim=stdev_coef_lim,
        trend=trend,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        legacy=legacy,
    )
    # 把 eb 里的字段回填到 fit
    fit.update({
        "df_prior": eb["df_prior"],
        "s2_prior": eb["s2_prior"],
        "var_prior": eb["var_prior"],
        "proportion": proportion,
        "s2_post": eb["s2_post"],
        "t": eb["t"],
        "df_total": eb["df_total"],
        "p_value": eb["p_value"],
        "lods": eb["lods"],
    })
    # 在 ebayes.py 中替换
    if fit.get("design") is not None and np.linalg.matrix_rank(fit["design"]) == fit["design"].shape[1]:
        F_stat = classify_tests_f(fit, fstat_only=True)  # 调用 classify_tests_f，只返回 F 统计量
        fit["F"] = F_stat.to_numpy().ravel()  # 将 F 统计量转换为一维数组
        df1 = F_stat.attrs["df1"]  # 获取分子自由度
        df2 = F_stat.attrs["df2"]  # 获取分母自由度
        fit["F_p_value"] = f.sf(fit["F"], df1, df2)  # 计算 F 统计量的 p 值
    return fit

def _ebayes(fit, proportion, stdev_coef_lim, trend, robust, winsor_tail_p, legacy):
    coefficients = fit["coefficients"]
    stdev_unscaled = fit["stdev_unscaled"]
    sigma = fit["sigma"]
    df_residual = fit["df_residual"]

    if (coefficients is None or
            stdev_unscaled is None or
            sigma is None or
            df_residual is None):
        raise ValueError("No data, or argument is not a valid lmFit object")

    if np.max(df_residual) == 0:
        raise ValueError("No residual degrees of freedom in linear model fits")

    if not np.any(np.isfinite(sigma)):
        raise ValueError("No finite residual standard deviations")

    sigma = fit["sigma"]

    if isinstance(trend, bool):
        if trend:
            covariate = fit.get("Amean")
            if covariate is None:
                raise ValueError("Need Amean component in fit to estimate trend")
        else:
            covariate = None
    elif isinstance(trend, (list, np.ndarray, pd.Series)):
        if len(trend) != len(sigma):
            raise ValueError("If trend is numeric then it should have length equal to the number of genes")
        covariate = trend
    else:
        raise ValueError("trend should be either a logical scale or a numeric vector")

    out = squeezeVar(sigma ** 2, df_residual, covariate=covariate, robust=robust, winsor_tail_p=winsor_tail_p, legacy=legacy)

    out["s2_prior"] = out["var_prior"]
    out["s2_post"] = out["var_post"]
    del out["var_prior"]
    del out["var_post"]

    # 1. 确保 s2_post 是形状为 (基因数, 1) 的二维数组
    s2_post = out["s2_post"].reshape(-1, 1)  # 关键：转换为列向量

    # 2. 将 coefficients 和 stdev_unscaled 转换为 numpy 数组（移除 pandas 索引影响）
    coef_np = coefficients.values
    stdev_np = stdev_unscaled.values

    # 3. 执行除法运算（此时形状兼容：[n×4] / [n×4] / [n×1]）
    out["t"] = coef_np / stdev_np / np.sqrt(s2_post)

    # 计算 df.total
    df_total = df_residual + out["df_prior"]

    # 计算 df.pooled（忽略 NaN 值）
    df_pooled = np.sum(df_residual, axis=0)

    # 确保 df.total 不超过 df.pooled
    df_total = np.minimum(df_total, df_pooled)

    out["df_total"] = df_total

    df_total_2d = np.asarray(out["df_total"])[:, np.newaxis]
    out["p_value"] = 2 * t.sf(np.abs(out["t"]), df=df_total_2d)

    stdev_coef_lim_sq = np.square(stdev_coef_lim)
    var_prior_lim = stdev_coef_lim_sq / np.median(out["s2_prior"])

    out["var_prior"] = tmixture_matrix(out["t"], stdev_unscaled, out["df_total"], proportion, var_prior_lim)

    if np.any(np.isnan(out["var_prior"])):
        # 将 NaN 值替换为 1 / s2_prior
        out["var_prior"][np.isnan(out["var_prior"])] = 1 / out["s2_prior"]
        # 发出警告
        import warnings
        warnings.warn("Estimation of var.prior failed - set to default value")

    r = np.tile(out["var_prior"], (stdev_unscaled.shape[1], 1)).T
    r = (stdev_unscaled ** 2 + r) / stdev_unscaled ** 2
    t2 = out["t"] ** 2

    Infdf = out["df_prior"] > 10 ** 6
    if np.any(Infdf):
        kernel = t2 * (1 - 1 / r) / 2
        if np.any(~Infdf):
            t2_f = t2[~Infdf]
            r_f = r[~Infdf]
            df_total_f = out["df_total"][~Infdf]
            kernel[~Infdf] = (1 + df_total_f) / 2 * np.log((t2_f + df_total_f) / (t2_f / r_f + df_total_f))
    else:
        # 在计算 kernel 之前，先处理 df.total 的形状
        df_total = out["df_total"]
        # 确保 df_total 是形状为 (10859, 1) 的二维数组
        df_total_2d = df_total.values.reshape(-1, 1)
        # 用调整后的 df_total_2d 进行计算
        kernel = (1 + df_total_2d) / 2 * np.log((t2 + df_total_2d) / (t2 / r + df_total_2d))

    out["lods"] = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel

    return out

def tmixture_matrix(tstat, stdev_unscaled, df, proportion, v0_lim=None):

    tstat = np.atleast_2d(tstat).T

    # 确保 stdev_unscaled 是二维数组
    stdev_unscaled = np.atleast_2d(stdev_unscaled).T

    if tstat.shape != stdev_unscaled.shape:
        raise ValueError("Dims of tstat and stdev.unscaled don't match")

    if v0_lim is not None and len(v0_lim) != 2:
        raise ValueError("v0_lim must have length 2")

    ncoef = tstat.shape[1]
    v0 = np.zeros(ncoef)
    for j in range(ncoef):
        v0[j] = tmixture_vector(tstat[:, j], stdev_unscaled[:, j], df, proportion, v0_lim)

    return v0

def tmixture_vector(tstat, stdev_unscaled, df, proportion, v0_lim=None):
    if np.any(np.isnan(tstat)):
        # 创建一个布尔掩码，表示非 NaN 的位置
        o = ~np.isnan(tstat)
        # 根据布尔掩码筛选数据
        tstat = tstat[o]
        stdev_unscaled = stdev_unscaled[o]
        df = df[o]

    ngenes = len(tstat)
    ntarget = int(np.ceil(proportion / 2 * ngenes))
    if ntarget < 1:
        return None

    p = max(ntarget / ngenes, proportion)
    tstat = np.abs(tstat)
    MaxDF = np.max(df)
    i = df < MaxDF
    if np.any(i):
        TailP = t.logsf(tstat[i], df[i])
        tstat[i] = t.logppf(TailP, MaxDF)
        df[i] = MaxDF

    o = np.argsort(-tstat)[:ntarget]
    tstat = tstat[o]
    v1 = stdev_unscaled[o] ** 2

    r = np.arange(1, ntarget + 1)
    p0 = 2 * t.sf(tstat, df=MaxDF)
    ptarget = ((r - 0.5) / ngenes - (1 - p) * p0) / p
    v0 = np.zeros(ntarget)
    pos = ptarget > p0
    if np.any(pos):
        qtarget = t.ppf(1 - ptarget[pos] / 2, df=MaxDF)
        v0[pos] = v1[pos] * ((tstat[pos] / qtarget) ** 2 - 1)

    if v0_lim is not None:
        v0 = np.clip(v0, v0_lim[0], v0_lim[1])

    return np.mean(v0)