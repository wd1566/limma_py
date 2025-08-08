import numpy as np
from scipy.special import digamma, polygamma
from scipy.linalg import lstsq
from patsy.highlevel import dmatrix

def logmdigamma(a):
    """Equivalent to R's logmdigamma: digamma(a) - log(a)"""
    return digamma(a) - np.log(a)


def trigammaInverse(x):
    """Exact reproduction of R's trigammaInverse function (修正停止条件)"""
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

    # 中间值用牛顿迭代法（与R的停止条件严格对齐）
    other = ~large & ~small
    if np.any(other):
        x_other = x_valid[other]
        y_other = 0.5 + 1.0 / x_other  # 初始值（与R一致）
        for _ in range(50):  # 最多50次迭代（与R一致）
            tri = polygamma(1, y_other)  # 三伽马函数（trigamma）
            tri2 = polygamma(2, y_other)  # 四伽马函数（psigamma(deriv=2)）
            dif = tri * (1 - tri / x_other) / tri2  # 迭代步长（与R公式一致）
            y_other += dif
            # 停止条件：与R完全一致（使用-dif而非绝对值）
            if np.max(-dif / y_other) < 1e-8:
                break
        y_valid[other] = y_other

    y[valid] = y_valid
    return y


def fitFDist(x, df1, covariate=None):
    """完整复刻R的fitFDist函数，支持协变量样条拟合和无效数据处理"""
    # 输入转换为numpy数组（确保处理一致性）
    x = np.asarray(x, dtype=np.float64)
    df1 = np.asarray(df1, dtype=np.float64)
    n = len(x)
    original_n = n  # 保留原始长度（用于处理无效数据后对齐）

    # 输入检查：空数据
    if n == 0:
        return {'scale': np.nan, 'df2': np.nan}
    # 输入检查：单数据点
    if n == 1:
        return {'scale': x[0], 'df2': 0}

    # 检查df1有效性
    if len(df1) == 1:
        # df1为单值时，检查是否有效
        ok_df1 = np.isfinite(df1) & (df1 > 1e-15)
        if not ok_df1:
            return {'scale': np.nan, 'df2': np.nan}
        ok = np.full(n, True)  # 单值df1有效则所有数据初始标记为有效
    else:
        # df1为向量时，需与x长度一致
        if len(df1) != n:
            raise ValueError("x and df1 have different lengths")
        ok = np.isfinite(df1) & (df1 > 1e-15)  # 标记df1有效的数据

    # 检查协变量长度（若存在）
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=np.float64)
        if len(covariate) != n:
            raise ValueError("x and covariate must be of same length")
        # 协变量预处理：处理无限值（与R逻辑一致）
        isfin_cov = np.isfinite(covariate)
        if not np.all(isfin_cov):
            if np.any(isfin_cov):
                # 用有限值的范围扩展无限值
                cov_range = [np.min(covariate[isfin_cov]), np.max(covariate[isfin_cov])]
                covariate[covariate == -np.inf] = cov_range[0] - 1
                covariate[covariate == np.inf] = cov_range[1] + 1
            else:
                # 全为无限值时用符号替代
                covariate = np.sign(covariate)

    # 过滤无效数据（x需有限且> -1e-15）
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = np.sum(ok)
    # 有效数据仅1个时直接返回
    if nok == 1:
        scale = np.full(original_n, x[ok][0])  # 扩展到原始长度
        return {'scale': scale, 'df2': 0}
    # 无有效数据时返回NA
    if nok == 0:
        return {'scale': np.nan, 'df2': np.nan}

    # 记录无效数据的协变量（用于后续预测）
    covariate_notok = None
    if covariate is not None:
        covariate_notok = covariate[~ok].copy()  # 无效数据的协变量

    # 过滤数据（仅保留有效部分）
    x = x[ok]
    if len(df1) > 1:
        df1 = df1[ok]
    if covariate is not None:
        covariate = covariate[ok]
    n = len(x)  # 更新n为有效数据量

    # 零值偏移处理（与R完全一致）
    m = np.median(x)
    if m == 0:
        m = 1  # 中位数为0时强制设为1
    x = np.maximum(x, 1e-5 * m)  # 偏移零值，避免log(x)无定义

    # 核心转换：计算e = log(x) - logmdigamma(df1/2)
    z = np.log(x)
    e = z - logmdigamma(df1 / 2)

    # 计算emean和evar（分有/无协变量两种情况）
    if covariate is None:
        # 无协变量：简单均值和样本方差
        emean = np.mean(e)
        evar = np.var(e, ddof=1)  # 无偏方差（除以n-1）
    else:
        # 有协变量：样条拟合（完整复刻R的逻辑）
        # 动态计算样条自由度（与R一致）
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, len(np.unique(covariate)))  # 不超过协变量唯一值数量

        # 若样条自由度不足，递归调用无协变量版本
        if splinedf < 2:
            result = fitFDist(x, df1, covariate=None)
            # 扩展结果到原始长度
            scale = np.full(original_n, result['scale'][0] if np.isscalar(result['scale']) else result['scale'][0])
            return {'scale': scale, 'df2': result['df2']}

        # 构建样条设计矩阵（类似R的splines::ns）
        try:
            # 用patsy的bs函数构建B样条设计矩阵（含截距项）
            design = dmatrix(f"bs(x, df={splinedf}, include_intercept=True)", {"x": covariate})
            design = np.asarray(design, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"样条设计矩阵构建失败: {e}")

        # 线性模型拟合（与R的lm.fit一致）
        fit_coef, residuals, rank, _ = lstsq(design, e, rcond=None)
        # 计算emean（拟合值）
        emean_fitted = design @ fit_coef

        # 计算evar（残差方差，与R的mean(fit$effects[-(1:fit$rank)]^2)一致）
        if rank < design.shape[1]:
            # 存在共线性时，去除冗余维度的残差
            evar = np.mean(residuals ** 2)
        else:
            evar = np.mean((e - emean_fitted) ** 2)

        # 预测无效数据的emean（与原始长度对齐）
        if covariate_notok is not None and len(covariate_notok) > 0:
            # 为无效数据构建样条设计矩阵
            design_notok = dmatrix(f"bs(x, df={splinedf}, include_intercept=True)", {"x": covariate_notok})
            design_notok = np.asarray(design_notok, dtype=np.float64)
            emean_notok = design_notok @ fit_coef  # 预测无效数据的emean
            # 合并有效和无效数据的emean（恢复原始顺序）
            emean_full = np.empty(original_n, dtype=np.float64)
            emean_full[ok] = emean_fitted
            emean_full[~ok] = emean_notok
            emean = emean_full
        else:
            # 无无效数据时直接用拟合值（扩展到原始长度）
            emean = np.full(original_n, np.nan)
            emean[ok] = emean_fitted

    # 调整evar：减去df1/2的三伽马函数均值（与R一致）
    evar = evar - np.mean(polygamma(1, df1 / 2))

    # 估计df2和scale
    if evar > 0:
        df2 = 2 * trigammaInverse(evar)
        # 处理df2为数组的情况（确保广播兼容）
        if np.isscalar(df2):
            s20 = np.exp(emean + logmdigamma(df2 / 2))
        else:
            s20 = np.exp(emean + logmdigamma(df2[:, np.newaxis] / 2))
    else:
        df2 = np.inf
        if covariate is None:
            # 无协变量时，scale为x的均值（与R一致）
            s20 = np.mean(x)
            s20 = np.full(original_n, s20)  # 扩展到原始长度
        else:
            # 有协变量时，scale为exp(emean)
            s20 = np.exp(emean)



    return {'scale': s20, 'df2': df2}