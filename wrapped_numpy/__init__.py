# cupy is only available on NVIDIA GPUs
# so import it only if it exists
import importlib.util

check_cupy = importlib.util.find_spec("cupy")

if check_cupy is not None:
    from wrapped_numpy.numba_wrapper.operations import *
    from wrapped_numpy.numba_wrapper.arr_operations import *
    from wrapped_numpy.numba_wrapper.literals import *

else:
    from wrapped_numpy.cupy_wrapper.operations import *
    from wrapped_numpy.cupy_wrapper.arr_operations import *
    from wrapped_numpy.cupy_wrapper.literals import *
