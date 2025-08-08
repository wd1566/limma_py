import numpy as np
from .fitFDist import fitFDist


def squeezeVar(var, df, covariate=None, robust=False, winsor_tail_p=(0.05, 0.1), legacy=None):
    var = np.asarray(var, dtype=float)  # 确保 float 类型
    n = var.size  # 基因数量

    # 无观测值时报错
    if n == 0:
        raise ValueError("var is empty")

    # 样本量<3时，经验贝叶斯无优势，直接返回原始方差
    if n < 3:
        return {"var.post": var,
                "var.prior": var,
                "df.prior": 0.0}

    df = np.asarray(df, dtype=float)

    # 处理无效自由度（df == 0）
    if df.size > 1:  # 等价于 R 的 length(df) > 1
        var[df == 0] = 0.0  # 将所有 df == 0 的位置对应的 var 置 0

    # 假设 df 已经是 np.ndarray(dtype=float)
    if legacy is None:  # 对应 R 的 is.null(legacy)
        dfp = df[df > 0]  # 仅保留有效自由度
        if dfp.size == 0:
            # 极端情况：没有有效自由度，统一用新方法
            legacy = False
        else:
            legacy = bool(np.min(dfp) == np.max(dfp))

    if legacy:  # 旧方法（df 相等）
        if robust:  # 稳健估计（抗异常值）
            raise NotImplementedError(
                "Robust F-distribution fitting (fitFDistRobustly) not implemented."
            )
        else:
            fit = fitFDist(var, df1=df, covariate=covariate)
            df_prior = fit['df2']
    else:  # 新方法（df 不相等）
        raise NotImplementedError(
            "Handling unequal degrees of freedom (fitFDistUnequalDF1) not implemented."
        )

    if np.any(np.isnan(df_prior)):
        raise ValueError("Could not estimate prior df")

    var_post = _squeezeVar(var=var, df=df,
                            var_prior=fit['scale'],
                            df_prior=df_prior)

    return {
        "df.prior": df_prior,
        "var.prior": fit["scale"],
        "var.post": var_post
    }

def _squeezeVar(var, df, var_prior, df_prior):
    m = np.max(df_prior)  # 先验自由度的最大值
    if np.isfinite(m):
        return (df * var + df_prior * var_prior) / (df + df_prior)

    n = var.size  # 基因数量
    if np.asarray(var_prior).size == n:
        var_post = var_prior
    else:
        var_post = np.resize(var_prior, n)

    m = np.min(df_prior)  # 先验自由度的最小值
    if m > 1e100:
        return var_post  # 视为无穷大，直接返回

    # 假设 var, df, df_prior, var_post 均已为同长度或广播后的 np.ndarray
    i = np.isfinite(df_prior)  # 布尔索引：先验自由度有限的基因
    if df.size > 1:  # 与 R 的 length(df) > 1 对应
        df = df[i]  # 提取对应残差自由度
    df_prior_finite = df_prior[i]  # 提取对应先验自由度

    # 仅对有限自由度的基因应用收缩公式
    var_post[i] = (df * var[i] + df_prior_finite * var_post[i]) / (df + df_prior_finite)

    return var_post