import numpy as np
import pandas as pd
from scipy import stats



def classify_tests_f(object, cor_matrix=None, df=None, p_value=0.01, fstat_only=False):
    if isinstance(object, dict):
        tstat = np.asarray(object["t"], dtype=float)
    else:
        tstat = np.asarray(object, dtype=float)
    tstat = pd.DataFrame(tstat)

    if isinstance(object, list):
        if object.get("t") is None:
            raise ValueError("tstat cannot be extracted from object")

    if cor_matrix is None and object.get("cov_coefficients") is not None:
        cov = object["cov_coefficients"]
        cov = np.asarray(cov)
        n = cov.shape[0]
        i = np.arange(n) + n * np.arange(n)
        if np.min(cov.flat[i]) == 0:
            j = i[cov.flat[i] == 0]
            cov.flat[j] = 1
        std = np.sqrt(np.diag(cov))
        cor_matrix = cov / np.outer(std, std)

    if df is None and object.get("df_prior") is not None and object.get("df_residual") is not None:
        df_prior = object["df_prior"]
        if np.isscalar(df_prior):
            df_prior_val = df_prior  # 标量直接使用
        else:
            # 数组取第一个元素并转换为Python标量
            df_prior_val = np.asarray(df_prior)[0].item()

        df_residual = object["df_residual"]
        if np.isscalar(df_residual):
            df_residual_val = df_residual  # 标量直接使用
        else:
            # 数组取第一个元素并转换为Python标量
            df_residual_val = np.asarray(df_residual)[0].item()

        df = float(df_prior_val + df_residual_val)
    if tstat.ndim == 1:
        tstat = tstat.reshape(-1, 1)
    else:
        if tstat.ndim == 1:
            tstat = tstat.reshape(-1, 1)

    ngenes = tstat.shape[0]  # 基因数量（行）
    ntests = tstat.shape[1]  # 检验数量（列）

    if ntests == 1:
        if fstat_only:
            fstat = tstat ** 2
            # 用自定义属性对象或直接返回 Series/DataFrame
            fstat = pd.Series(fstat.ravel())
            fstat.attrs["df1"] = 1
            fstat.attrs["df2"] = df
            return fstat
        else:
            p = 2 * stats.t.sf(np.abs(tstat), df)
            sign_t = np.sign(tstat)
            flag = sign_t * (p < p_value)
            return pd.DataFrame(flag, columns=["test"])

    if cor_matrix is None:
        r = ntests  # 相关矩阵为 None 时，有效秩 = 检验数
        Q = np.diag(np.ones(r)) / np.sqrt(r)  # 正交变换矩阵（默认对角线矩阵，标准化）
    else:
        E = np.linalg.eigh(cor_matrix)  # 对相关矩阵做特征值分解（对称矩阵）
        r = np.sum(E[0] / E[0][0] > 1e-08)  # 有效秩：特征值与最大特征值之比 > 1e-08 的数量
        Q = np.dot(E[1][:, :r], np.diag(1 / np.sqrt(E[0][:r]))) / np.sqrt(r)  # 构造变换矩阵

    if fstat_only:
        # 计算 F 统计量
        fstat = np.sum((tstat.values @ Q) ** 2, axis=1) # F 统计量 = 变换后 t 统计量的平方和
        # 设置属性
        fstat = pd.Series(fstat, index=tstat.index)  # 转换为 Series，保留索引
        fstat.attrs["df1"] = r  # 分子自由度 = 有效秩 r
        fstat.attrs["df2"] = df  # 分母自由度 = df
        return fstat

    qF = stats.f.ppf(1 - p_value, r, df)
    if qF.size == 1:
        qF = np.full(ngenes, qF)

    # 初始化结果矩阵，所有元素为 0
    result = np.zeros((ngenes, ntests))

    # 如果需要设置维度名称（行名和列名）
    if tstat.index is not None and tstat.columns is not None:
        result = pd.DataFrame(result, index=tstat.index, columns=tstat.columns)

    for i in range(ngenes):  # 循环每个基因
        x = tstat.iloc[i, :].values  # 提取该基因的 t 统计量
        if np.any(np.isnan(x)):  # 若存在 NaN，结果标记为 NaN
            result.iloc[i, :] = np.nan
        else:
            if np.dot(np.dot(Q, x), np.dot(Q, x)) > qF[i]:  # 变换后的 t 统计量平方和 > 临界值 qF：整体显著
                ord = np.argsort(np.abs(x))[::-1]  # 按 t 统计量绝对值从大到小排序
                result.iloc[i, ord[0]] = np.sign(x[ord[0]])  # 标记最显著检验的符号（±1）

                # 逐步判断次显著的检验
                for j in range(1, ntests):
                    bigger = ord[:j]  # 已标记的检验
                    x[bigger] = np.sign(x[bigger]) * np.abs(x[ord[j]])  # 将已标记检验的 t 值设为当前检验的绝对值（保持符号）
                    if np.dot(np.dot(Q, x), np.dot(Q, x)) > qF[i]:  # 若仍显著，标记当前检验的符号
                        result.iloc[i, ord[j]] = np.sign(x[ord[j]])
                    else:
                        break  # 若不显著，停止后续标记

    return result
