from numba import njit
import numpy as np
import numpy.typing as npt


@njit
def empty(shape: npt.NDArray[np.int_], dtype: npt.DTypeLike) -> npt.NDArray:
    return np.empty(shape=shape, dtype=dtype)


@njit
def empty_like(x: npt.ArrayLike) -> npt.NDArray:
    return np.empty_like(x)


@njit
def glorot_uniform(size: npt.NDArray[np.int_]) -> npt.NDArray:
    limit = (6 / (size[0] + size[1])) ** 0.5
    return np.random.uniform(low=-1.0 * limit, high=limit, size=size)


@njit
def linspace(start: float, stop: float, num: int) -> npt.NDArray:
    return np.linspace(start=start, stop=stop, num=num)


@njit
def zeros_like(x: npt.ArrayLike) -> npt.NDArray:
    return np.zeros_like(x)


@njit
def zeros(shape: npt.NDArray[np.int_], dtype: npt.DTypeLike) -> npt.NDArray:
    return np.zeros(shape=shape, dtype=dtype)
