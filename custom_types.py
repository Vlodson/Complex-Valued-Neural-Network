from typing import Dict, List, Tuple, TypedDict
import numpy as np
import numpy.typing as npt


type ComplexMatrix = npt.NDArray[np.complex_]
type NamedObjectDict[T] = Dict[str, T]
type CategoricalLabels = npt.NDArray[np.int_]
type MiniBatches = List[Tuple[ComplexMatrix, CategoricalLabels]]


class MetricInterResults(TypedDict):
    correct: int  # All correctly classified predictions for a single class
    instances: int  # All instances of a single class
    all_clf: int  # All classifications of a single class
