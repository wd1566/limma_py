import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from scipy.linalg import qr, solve_triangular
from scipy import stats
import warnings


def as_matrix_weights(weights, shape):
    """将权重转换为矩阵格式"""
    if weights is None:
        return None
    weights = np.array(weights)
    if weights.ndim == 1:
        if weights.shape[0] == shape[0]:
            # 行权重
            return np.tile(weights[:, np.newaxis], (1, shape[1]))
        elif weights.shape[0] == shape[1]:
            # 列权重
            return np.tile(weights[np.newaxis, :], (shape[0], 1))
        else:
            raise ValueError("权重维度不匹配")
    elif weights.shape == shape:
        return weights
    else:
        raise ValueError("权重维度不匹配")


def lm_series(M: np.ndarray,
              design: Optional[np.ndarray] = None,
              ndups: int = 1,
              spacing: int = 1,
              weights: Optional[np.ndarray] = None,
              debug: bool = True) -> Dict[str, Any]:
    """
    为每个基因拟合线性模型到一系列数组

    参数:
    M: 表达矩阵
    design: 设计矩阵
    ndups: 重复次数
    spacing: 间距
    weights: 权重
    debug: 是否输出调试信息

    返回:
    包含拟合结果的字典
    """

    # 检查表达矩阵
    M = np.asarray(M)
    narrays = M.shape[1]

    # 检查设计矩阵
    if design is None:
        design = np.ones((narrays, 1))
    else:
        design = np.asarray(design)

    nbeta = design.shape[1]
    coef_names = [f"x{i + 1}" for i in range(nbeta)]
    if hasattr(design, 'columns'):
        coef_names = list(design.columns)

    # 检查权重
    if weights is not None:
        weights = as_matrix_weights(weights, M.shape)
        weights[weights <= 0] = np.nan
        M[np.isnan(weights)] = np.nan

    # 将重复的行重新格式化为列
    if ndups > 1:
        M = unwrapdups(M, ndups=ndups, spacing=spacing)
        design = np.kron(design, np.ones(ndups))
        if weights is not None:
            weights = unwrapdups(weights, ndups=ndups, spacing=spacing)
        narrays = M.shape[1]

    # 初始化标准误
    ngenes = M.shape[0]
    stdev_unscaled = np.full((ngenes, nbeta), np.nan)
    beta = np.full((ngenes, nbeta), np.nan)

    # 设置行名和列名
    if hasattr(M, 'index'):
        row_names = list(M.index)
    else:
        row_names = [f"gene_{i}" for i in range(ngenes)]

    # 检查QR分解是否对所有基因都是常数
    NoProbeWts = np.all(np.isfinite(M)) and (weights is None or hasattr(weights, 'arrayweights'))

    if NoProbeWts:
        if weights is None:
            # 使用最小二乘拟合
            fit_coef, residuals, rank, s = np.linalg.lstsq(design, M.T, rcond=None)
            fit_coef = fit_coef.T
        else:
            # 使用加权最小二乘拟合
            W_sqrt = np.sqrt(weights[0, :])
            W_design = design * W_sqrt[:, np.newaxis]
            W_M = M.T * W_sqrt
            fit_coef, residuals, rank, s = np.linalg.lstsq(W_design, W_M, rcond=None)
            fit_coef = fit_coef.T

        # 计算sigma
        if rank < narrays:
            df_residual = narrays - rank
            if df_residual > 0:
                if residuals.ndim == 2:
                    sum_residuals = np.sum(residuals **2, axis=0)
                    sigma = np.sqrt(sum_residuals / df_residual)
                else:
                    sum_residuals = np.sum(residuals** 2)
                    sigma = np.sqrt(sum_residuals / df_residual)
            else:
                sigma = np.full(ngenes, np.nan)
        else:
            sigma = np.full(ngenes, np.nan)

        # 计算协方差矩阵
        try:
            Q, R = qr(design, mode='economic')
            cov_coef = np.linalg.inv(R.T @ R)
        except Exception as e:
            cov_coef = np.full((nbeta, nbeta), np.nan)

        # 计算未缩放的标准偏差
        est = np.arange(rank)
        diag_cov = np.sqrt(np.diag(cov_coef))
        stdev_unscaled[:, est] = np.tile(diag_cov, (ngenes, 1))

        df_residual_arr = np.full(ngenes, df_residual)

        result = {
            'coefficients': fit_coef,
            'stdev_unscaled': stdev_unscaled,
            'sigma': sigma,
            'df_residual': df_residual_arr,
            'cov_coefficients': cov_coef,
            'rank': rank
        }

        return result

    else:
        # 需要逐基因QR分解，因此遍历基因
        sigma = np.full(ngenes, np.nan)
        df_residual = np.zeros(ngenes)

        for i in range(ngenes):
            gene_name = row_names[i] if i < len(row_names) else f"gene_{i}"
            y = M[i, :].copy()
            obs = np.isfinite(y)
            valid_count = np.sum(obs)

            if valid_count > 0:
                X = design[obs, :]
                y_obs = y[obs]

                if weights is None:
                    # 普通最小二乘
                    try:
                        coef, residuals, rank, s = np.linalg.lstsq(X, y_obs, rcond=None)
                        beta[i, :] = coef

                        # 计算残差平方和
                        y_fitted = X @ coef
                        residuals = y_obs - y_fitted
                        sum_res = np.sum(residuals ** 2)

                        if rank > 0:
                            try:
                                Q, R = qr(X, mode='economic')
                                cov = np.linalg.inv(R.T @ R)
                                stdev_unscaled[i, :rank] = np.sqrt(np.diag(cov))
                            except Exception as e:
                                if debug:
                                    print(f"基因 {gene_name} QR分解错误: {e}")
                                pass

                        df_residual[i] = len(y_obs) - rank
                        if df_residual[i] > 0:
                            sigma[i] = np.sqrt(sum_res / df_residual[i])
                    except Exception as e:
                        if debug:
                            print(f"基因 {gene_name} 拟合错误: {e}")
                        pass

                else:
                    # 加权最小二乘
                    w = weights[i, obs]
                    W_sqrt = np.sqrt(w)
                    W_X = X * W_sqrt[:, np.newaxis]
                    W_y = y_obs * W_sqrt

                    try:
                        coef, residuals, rank, s = np.linalg.lstsq(W_X, W_y, rcond=None)
                        beta[i, :] = coef

                        # 计算加权残差平方和
                        y_fitted = X @ coef
                        residuals = y_obs - y_fitted
                        sum_res = np.sum(w * residuals ** 2)  # 使用权重加权的残差平方和

                        if rank > 0:
                            try:
                                Q, R = qr(W_X, mode='economic')
                                cov = np.linalg.inv(R.T @ R)
                                stdev_unscaled[i, :rank] = np.sqrt(np.diag(cov))
                            except Exception as e:
                                if debug:
                                    print(f"基因 {gene_name} 加权QR分解错误: {e}")
                                pass

                        df_residual[i] = len(y_obs) - rank
                        if df_residual[i] > 0:
                            sigma[i] = np.sqrt(sum_res / df_residual[i])
                    except Exception as e:
                        if debug:
                            print(f"基因 {gene_name} 加权拟合错误: {e}")
                        pass

    # 系数的相关矩阵
    try:
        Q, R = qr(design, mode='economic')
        cov_coef = np.linalg.inv(R.T @ R)
        rank = R.shape[1]
    except Exception as e:
        cov_coef = np.full((nbeta, nbeta), np.nan)
        rank = 0

    result = {
        'coefficients': beta,
        'stdev_unscaled': stdev_unscaled,
        'sigma': sigma,
        'df_residual': df_residual,
        'cov_coefficients': cov_coef,
        'rank': rank
    }

    return result


def nonEstimable(x: np.ndarray, tol: float = 1e-7) -> Optional[List[str]]:
    """
    检查设计矩阵中哪些系数是不可估计的

    参数:
    x: 设计矩阵
    tol: 容差值

    返回:
    不可估计的系数名称列表，如果所有系数都可估计则返回None
    """
    if x is None:
        return None

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    p = x.shape[1]  # 列数
    Q, R, pivots = qr(x, pivoting=True, mode='economic')

    # 计算秩
    rank = np.sum(np.abs(np.diag(R)) > tol)

    if rank == p:
        return None
    else:
        # 获取列名或生成默认列名
        if hasattr(x, 'columns'):
            n = list(x.columns)
        else:
            n = [f'coef_{i + 1}' for i in range(p)]

        # 找出不可估计的系数
        notest = [n[pivots[i]] for i in range(rank, p)]
        return notest


def uniquegenelist(probes: Any, ndups: int = 1, spacing: int = 1) -> Any:
    """
    从重复的探针列表中提取唯一的基因列表

    参数:
    probes: 探针数据，可以是向量、矩阵或数据框
    ndups: 重复次数
    spacing: 间距

    返回:
    唯一的基因列表
    """
    if ndups == 1:
        return probes

    if probes is None:
        return None

    if isinstance(probes, (list, np.ndarray)) and probes.ndim == 1:
        # 向量
        n = len(probes) // ndups
        return probes[:n]

    elif isinstance(probes, np.ndarray) and probes.ndim == 2:
        # 矩阵
        n = probes.shape[0] // ndups
        return probes[:n, :]

    elif hasattr(probes, 'iloc'):  # 类似pandas DataFrame
        n = len(probes) // ndups
        return probes.iloc[:n, :]

    else:
        raise ValueError("probes should be a vector, matrix or data.frame")


def unwrapdups(matrix: np.ndarray, ndups: int = 1, spacing: int = 1) -> np.ndarray:
    """
    将重复的数据展开为原始格式

    参数:
    matrix: 输入矩阵
    ndups: 重复次数
    spacing: 间距

    返回:
    展开后的矩阵
    """
    if ndups == 1:
        return matrix

    nrows = matrix.shape[0] * ndups
    if matrix.ndim == 1:
        result = np.zeros(nrows)
        for i in range(ndups):
            result[i::ndups] = matrix
    else:
        result = np.zeros((nrows, matrix.shape[1]))
        for i in range(ndups):
            result[i::ndups, :] = matrix

    return result


def mrlm(*args, **kwargs):
    """稳健回归功能未实现"""
    raise ValueError("稳健回归(mrlm)功能未实现")


# 添加一个简单的占位函数，用于处理广义最小二乘的情况
def gls_series(*args, **kwargs):
    """广义最小二乘功能未实现"""
    raise ValueError("广义最小二乘(gls.series)功能未实现")


def lmFit(object: Any,
          design: Optional[np.ndarray] = None,
          ndups: Optional[int] = None,
          spacing: Optional[int] = None,
          block: Optional[Any] = None,
          correlation: Optional[float] = None,
          weights: Optional[np.ndarray] = None,
          method: str = "ls",** kwargs) -> Dict[str, Any]:
    """
    为每个基因拟合线性模型

    参数:
    object: 输入数据对象，可以是数据框或包含表达数据的对象
    design: 设计矩阵
    ndups: 重复次数
    spacing: 间距
    block: 分组变量
    correlation: 相关性
    weights: 权重
    method: 方法，'ls'或'robust'
    **kwargs: 其他参数

    返回:
    包含拟合结果的字典
    """

    # 从输入对象中提取组件
    if isinstance(object, dict) and 'data' in object:
        # 处理数据框结构
        y = {'exprs': np.array(object['data'])}
        y['Amean'] = np.nanmean(y['exprs'], axis=1)
    elif hasattr(object, 'exprs'):
        # 假设对象有exprs属性
        y = {
            'exprs': object.exprs,
            'Amean': getattr(object, 'Amean', None),
            'probes': getattr(object, 'probes', None),
            'design': getattr(object, 'design', None),
            'weights': getattr(object, 'weights', None)
        }
        if hasattr(object, 'printer'):
            y['printer'] = {
                'ndups': getattr(object.printer, 'ndups', None),
                'spacing': getattr(object.printer, 'spacing', None)
            }
    else:
        # 假设object已经是包含必要字段的字典
        y = object

    if y['exprs'].shape[0] == 0:
        raise ValueError("表达矩阵有零行")

    # 检查设计矩阵
    if design is None:
        if 'design' in y:
            design = y['design']
        else:
            # 如果没有提供设计矩阵，默认为全1矩阵（仅截距模型）
            design = np.ones((y['exprs'].shape[1], 1))
    else:
        design = np.array(design)
        if design.dtype.kind not in 'iuf':
            raise ValueError("design必须是数值矩阵")
        if design.shape[0] != y['exprs'].shape[1]:
            raise ValueError("design的行维度与数据对象的列维度不匹配")
        if np.any(np.isnan(design)):
            raise ValueError("设计矩阵中不允许有NA值")

    # 检查哪些系数不可估计
    ne = nonEstimable(design)
    if ne is not None:
        print(f"不可估计的系数: {' '.join(ne)}")

    # 检查ndups和spacing，默认为1
    if ndups is None:
        if 'printer' in y and 'ndups' in y['printer']:
            ndups = y['printer']['ndups']
        else:
            ndups = 1

    if spacing is None:
        if 'printer' in y and 'spacing' in y['printer']:
            spacing = y['printer']['spacing']
        else:
            spacing = 1

    # 检查权重
    if weights is None and 'weights' in y:
        weights = y['weights']

    # 检查方法
    if method not in ["ls", "robust"]:
        raise ValueError("method必须是'ls'或'robust'")

    # 如果存在重复，将探针注释和Amean减少到正确的长度
    if ndups > 1:
        if 'probes' in y and y['probes'] is not None:
            y['probes'] = uniquegenelist(y['probes'], ndups=ndups, spacing=spacing)
        if 'Amean' in y and y['Amean'] is not None:
            unwrapped = unwrapdups(np.array(y['Amean']), ndups=ndups, spacing=spacing)
            y['Amean'] = np.nanmean(unwrapped, axis=0)

    if method == "robust":
        raise ValueError("稳健回归(mrlm)功能未实现")
    else:
        if ndups < 2 and block is None:
            # 使用最小二乘回归
            fit = lm_series(y['exprs'], design=design, ndups=ndups, spacing=spacing, weights=weights)

            fit['genes'] = y.get('probes', None)
            fit['Amean'] = y.get('Amean', None)
            fit['method'] = method
            fit['design'] = design
            fit['nonEstimable'] = ne

            return fit

        else:
            if correlation is None:
                raise ValueError("必须设置相关性，请参见duplicateCorrelation")
            raise ValueError("广义最小二乘(gls.series)功能未实现")

        # 关于缺失系数的可能警告
    if fit['coefficients'] is not None and fit['coefficients'].shape[1] > 1:
        n_missing = np.sum(np.isnan(fit['coefficients']), axis=1)
        n = np.sum((n_missing > 0) & (n_missing < fit['coefficients'].shape[1]))
        if n > 0:
            print(f"警告: 部分NA系数存在于{n}个探针中")

    # 模拟拟合结果
    fit = {
        'coefficients': np.zeros((y['exprs'].shape[0], design.shape[1])),
        'stdev_unscaled': np.ones((y['exprs'].shape[0], design.shape[1])),
        'sigma': np.ones(y['exprs'].shape[0]),
        'df_residual': np.full(y['exprs'].shape[0], y['exprs'].shape[1] - design.shape[1])
    }

    # 关于缺失系数的可能警告
    if fit['coefficients'].shape[1] > 1:
        n_missing = np.sum(np.isnan(fit['coefficients']), axis=1)
        n = np.sum((n_missing > 0) & (n_missing < fit['coefficients'].shape[1]))
        if n > 0:
            print(f"警告: 部分NA系数存在于{n}个探针中")

    # 输出结果
    result = {
        'genes': y.get('probes', None),
        'Amean': y.get('Amean', None),
        'method': method,
        'design': design,
        'coefficients': fit['coefficients'],  # 直接访问
        'stdev_unscaled': fit['stdev_unscaled'],  # 直接访问
        'sigma': fit['sigma'],  # 直接访问
        'df_residual': fit['df_residual'],  # 直接访问
        'cov_coefficients': fit.get('cov_coefficients', None),  # 这个可以用get，因为可能不存在
        'rank': fit.get('rank', None),  # 这个可以用get，因为可能不存在
        'nonEstimable': ne
    }

    return result
