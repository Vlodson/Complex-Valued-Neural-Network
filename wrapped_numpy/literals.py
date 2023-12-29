import numba
import numpy as np
import numpy.typing as npt


@numba.jit(nopython=True)
def empty(shape: npt.NDArray[np.int_], dtype: npt.DTypeLike) -> npt.NDArray:
    return np.empty(shape=shape, dtype=dtype)


@numba.jit(nopython=True)
def uniform(low: float, high: float, size: npt.NDArray[np.int_]) -> npt.NDArray:
    return np.random.uniform(low=low, high=high, size=size)


@numba.jit(nopython=True)
def linspace(start: float, stop: float, num: int) -> npt.NDArray:
    return np.linspace(start=start, stop=stop, num=num)


@numba.jit(nopython=True)
def zeros_like(x: npt.ArrayLike) -> npt.NDArray:
    return np.zeros_like(x)


@numba.jit(nopython=True)
def empty_like(x: npt.ArrayLike) -> npt.NDArray:
    return np.empty_like(x)
