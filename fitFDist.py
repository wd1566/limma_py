import numpy as np
from scipy.special import digamma, polygamma
from scipy.linalg import lstsq, qr
from ..utils.ns import ns


def logmdigamma(a):
    """Equivalent to R's logmdigamma: digamma(a) - log(a)"""
    return digamma(a) - np.log(a)


def trigammaInverse(x):
    """Exact reproduction of R's trigammaInverse function"""
    x = np.asarray(x)
    y = np.full_like(x, np.nan)
    valid = ~np.isnan(x) & (x >= 0)
    if not np.any(valid):
        return y

    x_valid = x[valid]
    y_valid = np.zeros_like(x_valid)

    # 处理大值 (x > 1e7)
    large = x_valid > 1e7
    y_valid[large] = 1.0 / np.sqrt(x_valid[large])

    # 处理小值 (x < 1e-6)
    small = x_valid < 1e-6
    y_valid[small] = 1.0 / x_valid[small]

    # 中间值用牛顿迭代法
    other = ~large & ~small
    if np.any(other):
        x_other = x_valid[other]
        y_other = 0.5 + 1.0 / x_other
        for _ in range(50):
            tri = polygamma(1, y_other)
            tri2 = polygamma(2, y_other)
            dif = tri * (1 - tri / x_other) / tri2
            y_other += dif
            if np.max(-dif / y_other) < 1e-8:
                break
        y_valid[other] = y_other

    y[valid] = y_valid
    return y


def fitFDist(x, df1, covariate=None):
    """完整复刻R的fitFDist函数"""
    # 输入转换为numpy数组
    x = np.asarray(x, dtype=np.float64)
    df1 = np.asarray(df1, dtype=np.float64)
    n = len(x)
    original_n = n

    # 输入检查
    if n == 0:
        return {'scale': np.nan, 'df2': np.nan}
    if n == 1:
        return {'scale': x[0], 'df2': 0}

    # 检查df1有效性
    if df1.size == 1:
        ok_df1 = np.isfinite(df1) & (df1 > 1e-15)
        if not ok_df1:
            return {'scale': np.nan, 'df2': np.nan}
        ok = np.full(n, True)
    else:
        if df1.size != n:
            raise ValueError("x and df1 have different lengths")
        ok = np.isfinite(df1) & (df1 > 1e-15)

    # 检查协变量
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=np.float64)
        if len(covariate) != n:
            raise ValueError("x and covariate must be of same length")
        # 处理无限值
        isfin_cov = np.isfinite(covariate)
        if not np.all(isfin_cov):
            if np.any(isfin_cov):
                cov_range = [np.min(covariate[isfin_cov]), np.max(covariate[isfin_cov])]
                covariate[covariate == -np.inf] = cov_range[0] - 1
                covariate[covariate == np.inf] = cov_range[1] + 1
            else:
                covariate = np.sign(covariate)

    # 过滤无效数据
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = np.sum(ok)
    if nok == 1:
        scale = np.full(original_n, x[ok][0])
        return {'scale': scale, 'df2': 0}
    if nok == 0:
        return {'scale': np.nan, 'df2': np.nan}

    # 记录无效数据的协变量
    covariate_notok = None
    if covariate is not None:
        covariate_notok = covariate[~ok].copy()

    # 过滤数据
    x = x[ok]
    if df1.size > 1:
        df1 = df1[ok]
    if covariate is not None:
        covariate = covariate[ok]
    n = len(x)

    # 零值偏移处理
    m = np.median(x)
    if m == 0:
        m = 1
    x = np.maximum(x, 1e-5 * m)

    # 核心转换
    z = np.log(x)
    e = z - logmdigamma(df1 / 2)

    # 计算emean和evar
    if covariate is None:
        emean = np.mean(e)
        evar = np.var(e, ddof=1)
    else:
        # 动态计算样条自由度
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, len(np.unique(covariate)))

        if splinedf < 2:
            result = fitFDist(x, df1, covariate=None)
            scale = np.full(original_n, result['scale'][0] if np.isscalar(result['scale']) else result['scale'][0])
            return {'scale': scale, 'df2': result['df2']}

        # 构建样条设计矩阵
        try:
            design = ns(x=covariate, df=splinedf, intercept=True)  # 使用 intercept=True
            design = np.asarray(design, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"样条设计矩阵构建失败: {e}")

        # 关键修复：使用QR分解来复刻R的lm.fit
        Q, R = qr(design, mode='economic')
        rank = np.linalg.matrix_rank(R)

        # 计算拟合系数
        fit_coef = lstsq(R, Q.T @ e, cond=None)[0]
        emean_fitted = design @ fit_coef

        # 关键修复：计算effects来复刻R的残差方差计算
        effects = Q.T @ e
        # 移除前rank个effects（与R的fit$effects[-(1:fit$rank)]一致）
        if rank < len(effects):
            residual_effects = effects[rank:]
            evar = np.mean(residual_effects ** 2)
        else:
            evar = np.mean((e - emean_fitted) ** 2)

        # 预测无效数据
        if covariate_notok is not None and len(covariate_notok) > 0:
            design_notok = ns(x=covariate_notok, df=splinedf, intercept=True)
            design_notok = np.asarray(design_notok, dtype=np.float64)
            emean_notok = design_notok @ fit_coef
            emean_full = np.empty(original_n, dtype=np.float64)
            emean_full[ok] = emean_fitted
            emean_full[~ok] = emean_notok
            emean = emean_full
        else:
            emean = np.full(original_n, np.nan)
            emean[ok] = emean_fitted

    evar = evar - np.mean(polygamma(1, df1 / 2))

    # 估计df2和scale
    if evar > 0:
        df2 = 2 * trigammaInverse(evar)
        s20 = np.exp(emean + logmdigamma(df2 / 2))
    else:
        df2 = np.inf
        if covariate is None:
            s20 = np.mean(x)
            s20 = np.full(original_n, s20)
        else:
            s20 = np.exp(emean)

    return {'scale': s20, 'df2': df2}
