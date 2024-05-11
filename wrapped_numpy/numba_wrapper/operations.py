from numba import njit
import numpy as np
import numpy.typing as npt


@njit(fastmath=True)
def absolute(x: npt.ArrayLike) -> npt.NDArray:
    return np.absolute(x)


@njit(fastmath=True)
def axis_sum(x: npt.ArrayLike, axis: int) -> npt.NDArray:
    return x.sum(axis=axis)


@njit(fastmath=True)
def sum_(x: npt.ArrayLike) -> npt.NDArray:
    return np.sum(x)


@njit(fastmath=True)
def abs_(x: npt.ArrayLike) -> npt.NDArray:
    return np.abs(x)


@njit(fastmath=True)
def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return a.dot(b)


@njit(fastmath=True)
def add(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return np.add(a, b)


@njit(fastmath=True)
def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return np.subtract(a, b)


@njit(fastmath=True)
def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return np.multiply(a, b)


@njit(fastmath=True)
def div(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return np.divide(a, b)


@njit(fastmath=True)
def pwr(x: npt.ArrayLike, exponent: npt.ArrayLike) -> npt.NDArray:
    return x**exponent


@njit
def transpose(x: npt.ArrayLike) -> npt.NDArray:
    return np.transpose(x)


@njit
def sign(x: npt.ArrayLike) -> npt.NDArray:
    return np.sign(x)


@njit(fastmath=True)
def exp(x: npt.ArrayLike) -> npt.NDArray:
    return np.exp(x)


@njit(fastmath=True)
def log(x: npt.ArrayLike) -> npt.NDArray:
    return np.log(x)


@njit(fastmath=True)
def sin_(x: npt.ArrayLike) -> npt.NDArray:
    return np.sin(x)


@njit(fastmath=True)
def cos_(x: npt.ArrayLike) -> npt.NDArray:
    return np.cos(x)


@njit(fastmath=True)
def tanh(x: npt.ArrayLike) -> npt.NDArray:
    return np.tanh(x)
