from numba import njit
import numpy as np
import numpy.typing as npt


@njit
def empty(shape: npt.NDArray[np.int_], dtype: npt.DTypeLike) -> npt.NDArray:
    return np.empty(shape=shape, dtype=dtype)


@njit
def uniform(low: float, high: float, size: npt.NDArray[np.int_]) -> npt.NDArray:
    return np.random.uniform(low=low, high=high, size=size)


@njit
def linspace(start: float, stop: float, num: int) -> npt.NDArray:
    return np.linspace(start=start, stop=stop, num=num)


@njit
def zeros_like(x: npt.ArrayLike) -> npt.NDArray:
    return np.zeros_like(x)


@njit
def empty_like(x: npt.ArrayLike) -> npt.NDArray:
    return np.empty_like(x)
