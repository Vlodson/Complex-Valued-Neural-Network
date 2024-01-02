from typing import Optional
import numba
import numpy as np
import numpy.typing as npt
from custom_types import ComplexMatrix


# only the base unique is wrapped
# any extra args like counts not included
@numba.jit(nopython=True)
def unique(x: npt.ArrayLike) -> npt.NDArray:
    return np.unique(x)


@numba.jit(nopython=True)
def angle(x: ComplexMatrix) -> npt.NDArray:
    return np.angle(x)


@numba.jit(nopython=True)
def real(x: ComplexMatrix) -> npt.NDArray:
    return np.real(x)


@numba.jit(nopython=True)
def imag(x: ComplexMatrix) -> npt.NDArray:
    return np.imag(x)


@numba.jit(nopython=True)
def where(condition: npt.ArrayLike, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
    return np.where(condition=condition, x=x, y=y)


# arraylike here being an arraylike of arraylikes
@numba.jit(nopython=True)
def concatenate(arrays: npt.ArrayLike, axis: Optional[int] = None) -> npt.NDArray:
    return (
        np.concatenate(arrays=arrays, axis=axis)
        if axis is not None
        else np.concatenate(arrays)
    )
