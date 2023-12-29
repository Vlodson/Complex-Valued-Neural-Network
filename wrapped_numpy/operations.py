import numba
import numpy as np
import numpy.typing as npt


@numba.jit(nopython=True)
def absolute(x: npt.ArrayLike) -> npt.NDArray:
    return np.absolute(x)


@numba.jit(nopython=True)
def axis_sum(x: npt.ArrayLike, axis: int) -> npt.NDArray:
    return np.sum(x, axis=axis)


@numba.jit(nopython=True)
def abs_(x: npt.ArrayLike) -> npt.NDArray:
    return np.abs(x)


@numba.jit(nopython=True)
def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return a @ b


@numba.jit(nopython=True)
def add(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    return a + b


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
def axis_max(x: npt.ArrayLike, axis: int) -> npt.NDArray:
    return np.max(x, axis=axis)


@numba.jit(nopython=True)
def tanh(x: npt.ArrayLike) -> npt.NDArray:
    return np.tanh(x)
