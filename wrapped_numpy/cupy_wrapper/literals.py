import cupy as cp
import numpy as np
import numpy.typing as npt


def empty(shape: npt.NDArray[np.int_], dtype: npt.DTypeLike) -> cp.ndarray:
    return cp.empty(shape=shape, dtype=dtype)


def empty_like(x: npt.ArrayLike) -> cp.ndarray:
    return cp.empty_like(x)


def glorot_uniform(size: npt.NDArray[np.int_]) -> cp.ndarray:
    limit = (6 / (size[0] + size[1])) ** 0.5
    return cp.random.uniform(low=-1.0 * limit, high=limit, size=size)


def linspace(start: float, stop: float, num: int) -> cp.ndarray:
    return cp.linspace(start=start, stop=stop, num=num)


def zeros_like(x: npt.ArrayLike) -> cp.ndarray:
    return cp.zeros_like(x)


def zeros(shape: npt.NDArray[np.int_], dtype: npt.DTypeLike) -> cp.ndarray:
    return cp.zeros(shape=shape, dtype=dtype)
