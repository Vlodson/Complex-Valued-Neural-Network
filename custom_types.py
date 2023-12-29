from typing import Dict, List, Tuple, TypedDict, TypeAlias, TypeVar
import numpy as np
import numpy.typing as npt


ComplexMatrix: TypeAlias = npt.NDArray[np.complex64]

T = TypeVar("T")
NamedObjectDict: TypeAlias = Dict[str, T]

CategoricalLabels: TypeAlias = npt.NDArray[np.int_]

MiniBatches: TypeAlias = List[Tuple[ComplexMatrix, CategoricalLabels]]


class MetricInterResults(TypedDict):
    correct: int  # All correctly classified predictions for a single class
    instances: int  # All instances of a single class
    all_clf: int  # All classifications of a single class
