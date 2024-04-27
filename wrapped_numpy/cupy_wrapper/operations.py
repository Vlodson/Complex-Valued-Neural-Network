import cupy as cp
import numpy.typing as npt


def absolute(x: npt.ArrayLike) -> cp.ndarray:
    return cp.absolute(cp.array(x))


def axis_sum(x: npt.ArrayLike, axis: int) -> cp.ndarray:
    return cp.sum(a=x, axis=axis)


def sum_(x: npt.ArrayLike) -> cp.ndarray:
    return cp.sum(x)


def abs_(x: npt.ArrayLike) -> cp.ndarray:
    return cp.abs(cp.array(x))


def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> cp.ndarray:
    return cp.dot(a, b)


def add(a: npt.ArrayLike, b: npt.ArrayLike) -> cp.ndarray:
    return cp.add(cp.array(a), cp.array(b))


def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> cp.ndarray:
    return cp.subtract(cp.array(a), cp.array(b))


def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> cp.ndarray:
    return cp.multiply(cp.array(a), cp.array(b))


def div(a: npt.ArrayLike, b: npt.ArrayLike) -> cp.ndarray:
    return cp.divide(cp.array(a), cp.array(b))


def pwr(x: npt.ArrayLike, exponent: npt.ArrayLike) -> cp.ndarray:
    return cp.power(cp.array(x), exponent)


def transpose(x: npt.ArrayLike) -> cp.ndarray:
    return cp.transpose(x)


def sign(x: npt.ArrayLike) -> cp.ndarray:
    return cp.sign(cp.array(x))


def exp(x: npt.ArrayLike) -> cp.ndarray:
    return cp.exp(cp.array(x))


def log(x: npt.ArrayLike) -> cp.ndarray:
    return cp.log(cp.array(x))


def sin_(x: npt.ArrayLike) -> cp.ndarray:
    return cp.sin(cp.array(x))


def cos_(x: npt.ArrayLike) -> cp.ndarray:
    return cp.cos(cp.array(x))


def tanh(x: npt.ArrayLike) -> cp.ndarray:
    return cp.tanh(cp.array(x))
