from typing import Optional
import numba
import numpy as np
import numpy.typing as npt


@numba.jit(nopython=True)
def absolute(x: npt.ArrayLike) -> npt.NDArray:
    return np.absolute(x)


@numba.jit(nopython=True)
def axis_sum(x: npt.ArrayLike, axis: Optional[int] = None) -> npt.NDArray:
    return np.sum(x, axis=axis) if axis is not None else np.sum(x)


@numba.jit(nopython=True)
def abs_(x: npt.ArrayLike) -> npt.NDArray:
    return np.abs(x)


@numba.jit(nopython=True)
def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return a.dot(b)


@numba.jit(nopython=True)
def add(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return np.add(a, b)


@numba.jit(nopython=True)
def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return np.subtract(a, b)


@numba.jit(nopython=True)
def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return np.multiply(a, b)


@numba.jit(nopython=True)
def div(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return np.divide(a, b)


@numba.jit(nopython=True)
def pwr(x: npt.ArrayLike, exponent: npt.ArrayLike) -> npt.NDArray:
    return x**exponent


@numba.jit(nopython=True)
def transpose(x: npt.ArrayLike) -> npt.NDArray:
    return np.transpose(x)


@numba.jit(nopython=True)
def sign(x: npt.ArrayLike) -> npt.NDArray:
    return np.sign(x)


@numba.jit(nopython=True)
def exp(x: npt.ArrayLike) -> npt.NDArray:
    return np.exp(x)


@numba.jit(nopython=True)
def log(x: npt.ArrayLike) -> npt.NDArray:
    return np.log(x)


@numba.jit(nopython=True)
def axis_max(x: npt.ArrayLike, axis: Optional[int] = None) -> npt.NDArray:
    return np.max(x, axis=axis) if x is not None else np.max(x)


@numba.jit(nopython=True)
def axis_min(x: npt.ArrayLike, axis: Optional[int] = None) -> npt.NDArray:
    return np.min(x, axis=axis) if x is not None else np.min(x)


@numba.jit(nopython=True)
def sin_(x: npt.ArrayLike) -> npt.NDArray:
    return np.sin(x)


@numba.jit(nopython=True)
def cos_(x: npt.ArrayLike) -> npt.NDArray:
    return np.cos(x)


@numba.jit(nopython=True)
def tanh(x: npt.ArrayLike) -> npt.NDArray:
    return np.tanh(x)
