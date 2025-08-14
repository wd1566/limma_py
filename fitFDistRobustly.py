import numpy as np
from scipy.stats import f as fdist
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from .fitFDist import fitFDist
from scipy.stats import chi2
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d


def fitFDistRobustly(x, df1, covariate=None, winsor_tail_p=(0.05, 0.1), trace=False):
    x = np.asarray(x)
    df1 = np.asarray(df1)
    n = len(x)

    if n < 2:
        return {'scale': None, 'df2': None, 'df2_shrunk': None}

    if n == 2:
        return fitFDist(x=x, df1=df1, covariate=covariate)

    if not (len(df1) == 1 or len(df1) == n):
        raise ValueError("x and df1 are different lengths")

    # Check covariate
    if covariate is not None:
        if len(covariate) != n:
            raise ValueError("x and covariate are different lengths")
        if not np.all(np.isfinite(covariate)):
            raise ValueError("covariate contains NA or infinite values")

    ok = ~np.isnan(x) & np.isfinite(df1) & (df1 > 1e-6)
    notallok = ~ok.all()

    if notallok:
        df2_shrunk = x.copy()  # 对应 R 的 df2.shrunk <- x

        # 1) 只保留 ok 对应的观测
        x = x[ok]
        if len(df1) > 1:
            df1 = df1[ok]
        if covariate is not None:
            covariate2 = covariate[~ok]  # 被剔除的协变量值
            covariate = covariate[ok]

        # 2) 递归调用
        fit = fitFDist(
            x=x,
            df1=df1,
            covariate=covariate,
            winsor_tail_p=winsor_tail_p,
            trace=trace
        )

        # 3) 回填结果
        df2_shrunk_out = df2_shrunk.copy()
        df2_shrunk_out[ok] = fit['df2_shrunk']  # fit$df2.shrunk
        df2_shrunk_out[~ok] = fit['df2']  # fit$df2

        # 4) scale 处理
        if covariate is None:
            scale = fit['scale']
        else:
            scale = df2_shrunk.copy()  # 先占位
            scale[ok] = fit['scale']

            # 近似插值：R 的 approx(..., rule = 2) -> interp1d(..., kind='linear', bounds_error=False, fill_value='extrapolate')
            log_scale_interp = interp1d(
                covariate,
                np.log(fit['scale']),
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            scale[~ok] = np.exp(log_scale_interp(covariate2))

        return {'scale': scale,
                'df2': fit['df2'],
                'df2_shrunk': df2_shrunk_out}

    m = np.median(x)
    if m <= 0:
        raise ValueError("Variances are mostly <= 0")

    i = x < m * 1e-12
    if np.any(i):
        x[i] = m * 1e-12

    NonRobust = fitFDist(x=x, df1=df1, covariate=covariate)

    prob = winsor_tail_p = np.resize(winsor_tail_p, 2)  # rep_len(..., 2L)
    prob[1] = 1 - winsor_tail_p[1]

    if np.all(winsor_tail_p < 1 / n):
        NonRobust['df2_shrunk'] = np.resize(NonRobust['df2'], n)
        return NonRobust

    if len(df1) > 1:
        df1max = df1.max()
        i = df1 < (df1max - 1e-14)
        if np.any(i):
            s = NonRobust['scale'] if covariate is None else NonRobust['scale'][i]
            f = x[i] / s
            df2 = NonRobust['df2']

            # 计算上、下尾 log-P 值
            pupper = fdist.logsf(f, dfn=df1[i], dfd=df2)  # log(P(X > f))
            plower = fdist.logcdf(f, dfn=df1[i], dfd=df2)  # log(P(X <= f))

            up = pupper < plower

            # 根据哪边更小，用逆 F 分布转换回新的 f
            if np.any(up):
                f[up] = fdist.isf(np.exp(pupper[up]), dfn=df1max, dfd=df2)
            if np.any(~up):
                f[~up] = fdist.ppf(np.exp(plower[~up]), dfn=df1max, dfd=df2)

            x[i] = f * s
            df1 = df1max
        else:
            df1 = df1[0]

    z = np.log(x)

    if covariate is None:
        # 计算截尾均值：winsor.tail.p[1] 对应 R 的 winsor.tail.p[1]
        trim_prop = winsor_tail_p[0]  # R 的 winsor.tail.p[2] 对应 winsor_tail_p[0]（索引 0）
        ztrend = np.mean(np.sort(z)[int(trim_prop * len(z)): int((1 - trim_prop) * len(z))])
        zresid = z - ztrend
    else:
        # loessFit(z, covariate, span = 0.4) -> statsmodels lowess
        lo = lowess(z, covariate, frac=0.4, return_sorted=False)
        ztrend = lo
        zresid = z - ztrend

    zrq = np.quantile(zresid, q=prob)  # 对应 R 的 quantile(..., probs = prob)
    zwins = np.clip(zresid, zrq[0], zrq[1])  # 等价于 pmin(pmax(...))

    zwmean = zwins.mean()
    zwvar = np.mean((zwins - zwmean) ** 2) * n / (n - 1)

    if trace:
        print(f"Variance of Winsorized Fisher-z {zwvar}")

    nodes_raw, weights_raw = np.polynomial.legendre.leggauss(128)
    g_nodes = (nodes_raw + 1) / 2.0  # 转换到 [0, 1]
    g_weights = weights_raw / 2.0  # 权重归一化（和为 1）

    linkfun = lambda x: x / (1 + x)
    linkinv = lambda x: x / (1 - x)

    def winsorized_moments(df1, df2, winsor_tail_p):
        # 计算尾部 F 分位数
        fq = fdist.ppf([winsor_tail_p[0], 1 - winsor_tail_p[1]], dfn=df1, dfd=df2)
        zq = np.log(fq)
        q = linkfun(fq)

        # 高斯积分节点与权重
        nodes = q[0] + (q[1] - q[0]) * g_nodes
        fnodes = linkinv(nodes)
        znodes = np.log(fnodes)

        # 被积函数：f(fnodes) / (1 - nodes)^2
        f_vals = fdist.pdf(fnodes, dfn=df1, dfd=df2) / ((1 - nodes) ** 2)

        q21 = q[1] - q[0]

        # 均值
        m = q21 * np.sum(g_weights * f_vals * znodes) + np.sum(zq * np.array(winsor_tail_p))

        # 方差
        v = q21 * np.sum(g_weights * f_vals * (znodes - m) ** 2) + np.sum((zq - m) ** 2 * np.array(winsor_tail_p))

        return {'mean': m, 'var': v}

    mom = winsorized_moments(df1=df1, df2=np.inf, winsor_tail_p=winsor_tail_p)
    funvalInf = np.log(zwvar / mom['var'])

    if funvalInf <= 0:
        df2 = np.inf

        # 2) 校正趋势以消除偏差
        ztrend_corrected = ztrend + zwmean - mom['mean']
        s20 = np.exp(ztrend_corrected)

        # 3) 计算异常值的尾部概率
        Fstat = np.exp(z - ztrend_corrected)
        TailP = chi2.sf(Fstat * df1, df=df1)  # 等价于 pchisq(..., lower.tail=FALSE)

        # 4) 经验尾部概率
        r = np.argsort(np.argsort(Fstat)) + 1  # 秩（R 的 rank）
        EmpiricalTailProb = (n - r + 0.5) / n

        # 5) 计算收缩后的自由度
        ProbNotOutlier = np.minimum(TailP / EmpiricalTailProb, 1.0)
        df_pooled = n * df1
        df2_shrunk = np.resize(df2, n)

        O = ProbNotOutlier < 1
        if np.any(O):
            df2_shrunk[O] = ProbNotOutlier[O] * df_pooled

            # 6) 按 TailP 升序排列，做累积最大值
            o = np.argsort(TailP)
            df2_shrunk[o] = np.maximum.accumulate(df2_shrunk[o])

        return {'scale': s20,
                'df2': df2,
                'tail.p.value': TailP,
                'df2_shrunk': df2_shrunk}

    def fun(x):
        df2 = linkinv(x)
        mom = winsorized_moments(df1=df1, df2=df2, winsor_tail_p=winsor_tail_p)
        if trace:
            print(f"df2= {df2}, Working Var= {mom['var']}")
        return np.log(zwvar / mom['var'])

    if NonRobust['df2'] == np.inf:
        NonRobust['df2_shrunk'] = np.resize(NonRobust['df2'], n)
        return NonRobust

    rbx = linkfun(NonRobust['df2'])
    funvalLow = fun(rbx)

    if funvalLow >= 0:
        df2 = NonRobust['df2']
    else:
        # 使用 SciPy 的 root_scalar 实现 uniroot
        sol = root_scalar(
            fun,
            bracket=[rbx, 1.0],
            x0=(rbx + 1.0) / 2.0,  # 可选：给出一个初始猜测
            xtol=1e-8
        )
        df2 = linkinv(sol.root)

    mom = winsorized_moments(df1=df1, df2=df2, winsor_tail_p=winsor_tail_p)
    ztrend_corrected = ztrend + zwmean - mom['mean']
    s20 = np.exp(ztrend_corrected)

    zresid = z - ztrend_corrected
    Fstat = np.exp(zresid)

    # 计算 log 尾部 P 值
    LogTailP = fdist.logsf(Fstat, dfn=df1, dfd=df2)  # log(P(X > Fstat))
    TailP = np.exp(LogTailP)

    # 经验尾部概率（对数形式）
    r = np.argsort(np.argsort(Fstat)) + 1  # 秩
    LogEmpiricalTailProb = np.log(n - r + 0.5) - np.log(n)

    # ProbNotOutlier 及其补数
    LogProbNotOutlier = np.minimum(LogTailP - LogEmpiricalTailProb, 0)
    ProbNotOutlier = np.exp(LogProbNotOutlier)
    ProbOutlier = -np.expm1(LogProbNotOutlier)

    if np.any(LogProbNotOutlier < 0):
        # 计算 df2.outlier
        minLogTailP = LogTailP.min()
        if np.isneginf(minLogTailP):
            df2_outlier = 0.0
            df2_shrunk = ProbNotOutlier * df2
        else:
            df2_outlier = np.log(0.5) / minLogTailP * df2

            newLogTailP = fdist.logsf(Fstat.max(), dfn=df1, dfd=df2_outlier)
            df2_outlier = np.log(0.5) / newLogTailP * df2_outlier

            df2_shrunk = ProbNotOutlier * df2 + ProbOutlier * df2_outlier

        # 强制按 TailP 单调递增
        o = np.argsort(LogTailP)  # 升序索引
        df2_ordered = df2_shrunk[o]
        m = np.cumsum(df2_ordered) / np.arange(1, n + 1)
        imin = np.argmin(m)
        df2_ordered[:imin] = m[imin]
        df2_shrunk[o] = np.maximum.accumulate(df2_ordered)

    else:
        df2_outlier = df2
        df2_shrunk = np.resize(df2, n)

    return {
        'scale': s20,
        'df2': df2,
        'tail.p.value': TailP,
        'prob.outlier': ProbOutlier,
        'df2.outlier': df2_outlier,
        'df2_shrunk': df2_shrunk
    }
