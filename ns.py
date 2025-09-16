import numpy as np
import pandas as pd
import warnings
from scipy.linalg import qr
from scipy.interpolate import BSpline


def spline_design(knots, x, ord=4, derivs=0):
    """
    完全复刻R的splineDesign函数
    """
    if not isinstance(derivs, (list, tuple, np.ndarray)):
        derivs = [derivs]

    n = len(knots) - ord
    result = np.zeros((len(x), n))

    for i in range(n):
        # 创建B样条基函数
        coefs = np.zeros(n)
        coefs[i] = 1.0
        spl = BSpline(knots, coefs, ord - 1, extrapolate=False)

        for j, deriv in enumerate(derivs):
            if deriv == 0:
                result[:, i] = spl(x)
            else:
                # 计算导数
                spl_deriv = spl.derivative(deriv)
                result[:, i] = spl_deriv(x)

    return result


def ns(x, df=None, knots=None, intercept=False, Boundary_knots=None):
    """
    完全复刻R的splines::ns函数
    """
    # 保存名称
    nx = None
    if hasattr(x, 'name'):
        nx = x.name
    elif hasattr(x, 'index'):
        nx = x.index

    x = np.asarray(x).flatten()
    nax = np.isnan(x)
    nas = nax.any()
    if nas:
        x = x[~nax]

    # 处理边界节点 - 完全按照R的逻辑
    if Boundary_knots is None:
        Boundary_knots = np.array([np.min(x), np.max(x)])
    else:
        Boundary_knots = np.sort(Boundary_knots)

    # 检查是否在边界外
    outside = (x < Boundary_knots[0]) | (x > Boundary_knots[1])

    # 确定内部节点 - 完全按照R的逻辑
    mk_knots = (df is not None and knots is None)
    if mk_knots:
        nIknots = df - 1 - intercept
        if nIknots < 0:
            nIknots = 0
            warnings.warn(f"'df' was too small; have used {1 + intercept}")

        if nIknots > 0:
            # 完全按照R的quantile计算方式
            knots_pos = np.linspace(0, 1, nIknots + 2)[1:-1]
            # 使用R类型的quantile计算
            x_filtered = x[~outside]
            knots = np.quantile(x_filtered, knots_pos,
                                method='linear')  # R默认使用type=7
        else:
            knots = None
    else:
        if knots is not None and not np.all(np.isfinite(knots)):
            raise ValueError("non-finite knots")
        nIknots = len(knots) if knots is not None else 0

    # 调整与边界节点重合的内部节点 - 完全按照R的逻辑
    if mk_knots and knots is not None and len(knots) > 0:
        lr_eq = [np.min(knots) == Boundary_knots[0], np.max(knots) == Boundary_knots[1]]

        if lr_eq[0]:
            piv = Boundary_knots[0]
            i = knots == piv
            if np.all(i):
                raise ValueError("all interior knots match left boundary knot")
            knots[i] = knots[i] + (np.min(knots[knots > piv]) - piv) / 8

        if lr_eq[1]:
            piv = Boundary_knots[1]
            i = knots == piv
            if np.all(i):
                raise ValueError("all interior knots match right boundary knot")
            knots[i] = knots[i] - (piv - np.max(knots[knots < piv])) / 8

        if any(lr_eq):
            warnings.warn("shoving 'interior' knots matching boundary knots to inside")

    # 构建扩展的节点序列 - 完全按照R的逻辑
    if knots is None:
        Aknots = np.repeat(Boundary_knots, 4)
    else:
        Aknots = np.sort(np.concatenate([np.repeat(Boundary_knots, 4), knots]))

    # 计算样条基函数 - 使用R的逻辑
    if np.any(outside):
        basis = np.zeros((len(x), len(Aknots) - 4))

        # 处理左边界的点
        if np.any(ol := (x < Boundary_knots[0])):
            k_pivot = Boundary_knots[0]
            xl = x[ol]

            # 计算在边界点处的函数值和一阶导数
            tt0 = spline_design(Aknots, [k_pivot], ord=4, derivs=0)
            tt1 = spline_design(Aknots, [k_pivot], ord=4, derivs=1)
            tt = np.vstack([tt0, tt1])

            # 线性外推 - 完全按照R的逻辑
            xl_mat = np.column_stack([np.ones(len(xl)), xl - k_pivot])
            basis[ol, :] = xl_mat @ tt

        # 处理右边界的点
        if np.any(or_ := (x > Boundary_knots[1])):
            k_pivot = Boundary_knots[1]
            xr = x[or_]

            # 计算在边界点处的函数值和一阶导数
            tt0 = spline_design(Aknots, [k_pivot], ord=4, derivs=0)
            tt1 = spline_design(Aknots, [k_pivot], ord=4, derivs=1)
            tt = np.vstack([tt0, tt1])

            # 线性外推 - 完全按照R的逻辑
            xr_mat = np.column_stack([np.ones(len(xr)), xr - k_pivot])
            basis[or_, :] = xr_mat @ tt

        # 处理内部点
        if np.any(inside := ~outside):
            basis[inside, :] = spline_design(Aknots, x[inside], ord=4)
    else:
        basis = spline_design(Aknots, x, ord=4)

    # 计算约束矩阵（二阶导数在边界点处） - 完全按照R的逻辑
    const = spline_design(Aknots, Boundary_knots, ord=4, derivs=2)

    # 移除截距项（如果需要） - 完全按照R的逻辑
    if not intercept:
        const = const[:, 1:]
        basis = basis[:, 1:]

    # QR分解和正交投影 - 完全按照R的逻辑
    const_t = const.T
    Q, R = qr(const_t, mode='full')

    # 计算 qr.qty(qr.const, t(basis))
    basis_t = basis.T
    basis_transformed = Q.T @ basis_t

    # 转置回来并移除前两列 - 完全按照R的逻辑
    basis_result = basis_transformed.T
    if basis_result.shape[1] > 2:
        basis_result = basis_result[:, 2:]
    else:
        basis_result = np.zeros((basis_result.shape[0], 0))

    # 处理缺失值
    if nas:
        nmat = np.full((len(nax), basis_result.shape[1]), np.nan)
        nmat[~nax, :] = basis_result
        basis_result = nmat

    # 创建结果DataFrame
    n_col = basis_result.shape[1]
    columns = [str(i + 1) for i in range(n_col)]

    result = pd.DataFrame(basis_result, columns=columns)

    # 添加元数据（与R保持一致）
    result._metadata = ['degree', 'knots', 'Boundary_knots', 'intercept']
    result.degree = 3
    result.knots = np.array([]) if knots is None else knots
    result.Boundary_knots = Boundary_knots
    result.intercept = intercept

    return result
