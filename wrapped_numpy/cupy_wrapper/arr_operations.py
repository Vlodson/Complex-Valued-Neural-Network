from typing import Optional
import cupy as cp
import numpy.typing as npt
from custom_types import ComplexMatrix


def unique(x: npt.ArrayLike) -> cp.ndarray:
    return cp.unique(x)


def angle(x: ComplexMatrix) -> cp.ndarray:
    # ndarray is not allowed in some functions
    # so cast it to cp array
    # luckily np can work with cp arrays
    return cp.angle(cp.array(x))


def real(x: ComplexMatrix) -> cp.ndarray:
    return cp.real(x)


def imag(x: ComplexMatrix) -> cp.ndarray:
    return cp.imag(x)


def where(condition: npt.ArrayLike, x: npt.ArrayLike, y: npt.ArrayLike) -> cp.ndarray:
    return cp.where(condition=cp.array(condition), x=cp.array(x), y=cp.array(y))


def concatenate(arrays: npt.ArrayLike, axis: Optional[int] = None) -> cp.ndarray:
    return (
        cp.concatenante(tup=arrays, axis=axis)
        if axis is not None
        else cp.concantenate(tup=arrays)
    )
