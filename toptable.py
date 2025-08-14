import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import statsmodels.stats.multitest as smm
from statsmodels.stats.multitest import multipletests


def toptable(fit,
             coef=None,
             number=10,
             genelist=None,
             adjust_method="BH",
             sort_by="B",
             resort_by=None,
             p_value=1.0,
             fc=None,
             lfc=None,
             confint=False):
    if genelist is None:
        genelist = pd.DataFrame(index=fit["coefficients"].index)

    if not isinstance(fit, dict):
        raise TypeError("fit must be a dict-like object")

    if "coefficients" not in fit or fit["coefficients"] is None:
        raise ValueError("coefficients not found in fit object")

    t_exists = "t" in fit and fit["t"] is not None
    F_exists = "F" in fit and fit["F"] is not None
    if not (t_exists or F_exists):
        raise ValueError("Need to run eBayes or treat first")

    if confint:
        if "stdev.unscaled" not in fit or fit["stdev.unscaled"] is None:
            raise ValueError("stdev.unscaled not found in fit object")

    if coef is None:
        if fit.get("treat_lfc") is None:
            coef = list(range(fit["coefficients"].shape[1]))
            cn = fit["design"].columns.tolist()
            if cn is not None:
                try:
                    i = cn.index("(Intercept)")
                    coef.pop(i)
                    print("Removing intercept from test coefficients")
                except ValueError:
                    pass
        else:
            coef = fit["coefficients"].shape[1] - 1

    if adjust_method is None:
        adjust_method = "BH"


    if fc is None:
        if lfc is None:
            lfc = 0
    else:
        if fc < 1:
            raise ValueError("fc must be greater than or equal to 1")
        lfc = np.log2(fc)

    if len(coef) > 1:
        if fit.get("treat.lfc") is not None:
            raise ValueError("Treat p-values can only be displayed for single coefficients")

        coef = list(dict.fromkeys(coef))  # unique，保持顺序

        # 如果当前 fit 的系数列数 > 要保留的列数，就按 coef 切片
        if fit["coefficients"].shape[1] > len(coef):
            fit = {k: v.iloc[:, coef] if k in ("coefficients", "t", "p_value", "lods", "stdev_unscaled")
            else v for k, v in fit.items()}

        if sort_by == "B":
            sort_by = "F"

        # 调用 .topTableF 的 Python 版本
        return _topTableF(fit,
                            number=number,
                            genelist=genelist,
                            adjust_method=adjust_method,
                            sort_by=sort_by,
                            p_value=p_value,
                            lfc=lfc)
    # 1. 把 fit 变成普通 dict/DataFrame（相当于 R 的 unclass）
    fit = dict(fit)  # 如果 fit 原本是自定义对象，转成纯 dict

    # 2. 构造 ebcols
    ebcols = ["t", "p_value", "lods"]
    if confint:
        ebcols = ["s2_post", "df_total"] + ebcols

    # 3. 组装传给 _top_table_t 的参数
    return _topTableT(
        fit={
            "coefficients": fit["coefficients"].iloc[:, coef],  # 只保留 coef 列
            "stdev_unscaled": fit["stdev_unscaled"].iloc[:, coef]
        },
        coef=coef,
        number=number,
        genelist=genelist,
        A=fit.get("Amean"),  # 如果 fit 里存的是 Amean
        eb={k: fit[k] for k in ebcols if k in fit},
        adjust_method=adjust_method,
        sort_by=sort_by,
        resort_by=resort_by,
        p_value=p_value,
        lfc=lfc,
        confint=confint
    )


def _topTableF(fit,
               number=10,
               genelist=None,
               adjust_method="BH",
               sort_by="F",
               p_value=1,
               lfc=0):
    if fit.get("coefficients") is None:
        raise ValueError("Coefficients not found in fit")

    M = np.asarray(fit["coefficients"])
    rn = fit["coefficients"].index.tolist()

    if rn is None or len(rn) == 0:
        rn = [f"Gene{i}" for i in range(M.shape[0])]

    if fit["coefficients"].columns.name is None or len(fit["coefficients"].columns) == 0:
        col_names = [f"Coef{i + 1}" for i in range(M.shape[1])]
    else:
        col_names = fit["coefficients"].columns.tolist()

    Amean = fit.get("Amean")
    Fstat = fit.get("F")
    Fp = fit.get("F_p_value")

    if Fstat is None:
        raise ValueError("F-statistics not found in fit")

    if genelist is not None and (not isinstance(genelist, pd.DataFrame) or genelist.ndim == 1):
        genelist = pd.DataFrame({"ProbeID": genelist})

    # 处理行名
    if rn is None or len(rn) == 0:
        rn = list(range(M.shape[0]))
    else:
        # 检查重复行名
        if len(rn) != len(set(rn)):
            if genelist is None:
                genelist = pd.DataFrame({"ID": rn})
            else:
                if "ID" in genelist.columns:
                    genelist = genelist.assign(ID0=rn)
                else:
                    genelist = genelist.assign(ID=rn)
            rn = list(range(M.shape[0]))  # 用数字索引替换

    valid = ("F", "none")
    if sort_by not in valid:
        raise ValueError(f"sort.by must be one of {valid}")

    _method_map = {"BH": "fdr_bh", "BY": "fdr_by",
                   "bonferroni": "bonferroni", "holm": "holm", "none": "none"}
    adjust_method = _method_map.get(adjust_method, adjust_method)
    adj_p_value = smm.multipletests(Fp, method=adjust_method)[1]

    if lfc > 0 or p_value < 1:
        # 倍数过滤
        if lfc > 0:
            big = np.any(np.abs(M) > lfc, axis=1)  # 每行至少一个系数满足
        else:
            big = np.ones(M.shape[0], dtype=bool)

        # p 值过滤
        if p_value < 1:
            sig = adj_p_value <= p_value
            sig = np.where(np.isnan(sig), False, sig)
        else:
            sig = np.ones(M.shape[0], dtype=bool)

        keep = big & sig

        # 按索引统一切片 - 关键修复点：使用iloc处理pandas对象
        if not keep.all():
            idx = np.where(keep)[0]
            M = M[idx, :]
            rn = [rn[i] for i in idx]

            # 对pandas Series使用iloc进行位置索引
            if isinstance(Amean, pd.Series):
                Amean = Amean.iloc[idx]
            else:
                Amean = Amean[idx]

            if isinstance(Fstat, pd.Series):
                Fstat = Fstat.iloc[idx]
            else:
                Fstat = Fstat[idx]

            if isinstance(Fp, pd.Series):
                Fp = Fp.iloc[idx]
            else:
                Fp = Fp[idx]

            genelist = genelist.iloc[idx] if genelist is not None else None
            adj_p_value = adj_p_value[idx]

    # 确保number不超过可用数据量
    if M.shape[0] < number:
        number = M.shape[0]
    if number < 1:
        return pd.DataFrame()

    # 排序索引
    if sort_by == "F":
        o = np.argsort(Fp)[:number]  # 升序（小 p 值在前）
    else:
        o = np.arange(number)

    # 构建结果表格
    if genelist is None:
        tab = pd.DataFrame(M[o, :], columns=col_names)
    else:
        # 确保genelist切片正确
        tab = genelist.iloc[o, :].copy()
        m_df = pd.DataFrame(M[o, :], columns=col_names, index=tab.index)  # 确保索引一致
        tab = pd.concat([tab, m_df], axis=1)

    # 添加统计量列 - 确保所有数组长度一致
    tab["AveExpr"] = Amean.iloc[o].to_numpy() if isinstance(Amean, pd.Series) else Amean[o]
    tab["F"] = Fstat.iloc[o].to_numpy() if isinstance(Fstat, pd.Series) else Fstat[o]
    tab["P.Value"] = Fp.iloc[o].to_numpy() if isinstance(Fp, pd.Series) else Fp[o]
    tab["adj.P.Val"] = adj_p_value[o]

    # 设置索引
    tab.index = [rn[i] for i in o]

    return tab


def _topTableT(fit,
               coef=1,
               number=10,
               genelist=None,
               A=None,
               eb=None,
               adjust_method="BH",
               sort_by="B",
               resort_by=None,
               p_value=1,
               lfc=0,
               confint=False):
    fit["coefficients"] = np.asarray(fit["coefficients"])
    rn = fit["coefficients"].index.tolist()

    if len(coef) > 1:
        coef = coef[0]
        import warnings
        warnings.warn("Treat is for single coefficients: only first value of coef being used")

    if genelist is not None and (not isinstance(genelist, pd.DataFrame) or genelist.ndim == 1):
        genelist = pd.DataFrame({"ID": genelist})

    # Check rownames
    rn = fit["coefficients"].index.tolist()
    if rn is None or len(rn) == 0:
        rn = list(range(fit["coefficients"].shape[0]))
    else:
        # Python 检测重复
        if len(rn) != len(set(rn)):
            if genelist is None:
                genelist = pd.DataFrame({"ID": rn})
            else:
                if "ID" in genelist.columns:
                    genelist["ID0"] = rn
                else:
                    genelist["ID"] = rn
            rn = list(range(fit["coefficients"].shape[0]))

    valid = ["logFC", "M", "A", "Amean", "AveExpr", "P", "p", "T", "t", "B", "none"]
    if sort_by not in valid:
        raise ValueError(f"sort.by must be one of {valid}")

    # 别名转换
    alias_map = {
        "M": "logFC",
        "A": "AveExpr",
        "Amean": "AveExpr",
        "T": "t",
        "p": "P"
    }
    sort_by = alias_map.get(sort_by, sort_by)  # 映射或保持原值

    if resort_by is not None:
        valid_res = ["logFC", "M", "A", "Amean", "AveExpr", "P", "p", "T", "t", "B"]
        if resort_by not in valid_res:
            raise ValueError(f"resort.by must be one of {valid_res}")

        alias_res = {"M": "logFC", "A": "AveExpr", "Amean": "AveExpr",
                     "p": "P", "T": "t"}
        resort_by = alias_res.get(resort_by, resort_by)

    if A is None:

        if sort_by == "AveExpr":
            raise ValueError("Cannot sort by A-values as these have not been given")
    else:
        A = np.asarray(A)
        if A.ndim > 1:
            A = np.nanmean(A, axis=1)

    # Check for lods component
    if eb is None or eb.get("lods") is None:
        if sort_by == "B":
            raise ValueError("Trying to sort.by B, but B-statistic (lods) not found")
        if resort_by is not None and resort_by == "B":
            raise ValueError("Trying to resort.by B, but B-statistic (lods) not found")
        include_B = False
    else:
        include_B = True

    # 提取指定列（coef 已转成 0-based 整数）
    M = fit["coefficients"][:, coef]  # logFC
    tstat = np.asarray(eb["t"])[:, coef]  # t 统计量
    P_Value = np.asarray(eb["p_value"])[:, coef]  # p 值
    if include_B:
        B = np.asarray(eb["lods"])[:, coef]  # B 统计量（lods）
    else:
        B = None

    adj_P_Value = smm.multipletests(P_Value, method=adjust_method)[1]

    # 根据 p.value 和 lfc 阈值过滤
    mask = (adj_P_Value <= p_value) & (np.abs(M) >= lfc)
    mask = np.where(np.isnan(mask), False, mask)

    if not mask.any():
        return pd.DataFrame()

    idx = np.where(mask)[0]

    genelist = genelist.iloc[idx] if genelist is not None else None
    M = M[idx]
    A = A[idx] if A is not None else None
    tstat = tstat[idx]
    P_Value = P_Value[idx]
    adj_P_Value = adj_P_Value[idx]
    if include_B:
        B = B[idx]
    rn = [rn[i] for i in idx]

    if len(M) < number:
        number = len(M)
    if number < 1:
        return pd.DataFrame()

    # 生成排序下标
    if sort_by == "logFC":
        order = np.argsort(-np.abs(M))
    elif sort_by == "AveExpr":
        order = np.argsort(-A)
    elif sort_by == "P":
        order = np.argsort(P_Value)
    elif sort_by == "t":
        order = np.argsort(-np.abs(tstat))
    elif sort_by == "B":
        order = np.argsort(-B)
    else:  # none
        order = np.arange(len(M))

    top = order[:number]

    if genelist is None:
        tab = pd.DataFrame({"logFC": M[top]})
    else:
        tab = genelist.iloc[top].copy()
        tab["logFC"] = M[top]

    # 2) 置信区间
    if confint:
        alpha = 0.975 if not isinstance(confint, (int, float)) else (1 + float(confint)) / 2
        df_t = eb["df_total"][top]
        se = np.sqrt(eb["s2_post"][top]) * fit["stdev_unscaled"][top, coef]
        margin = se * t_dist.ppf(alpha, df_t)
        tab["CI.L"] = M[top] - margin
        tab["CI.R"] = M[top] + margin

    # 3) 其余统计量
    if A is not None:
        tab["AveExpr"] = A[top]

    tab = pd.concat([
        tab,
        pd.DataFrame({
            "t": tstat[top],
            "P.Value": P_Value[top],
            "adj.P.Val": adj_P_Value[top]
        }, index=tab.index)
    ], axis=1)

    if include_B:
        tab["B"] = B[top]

    tab.index = [rn[i] for i in top]
    return tab