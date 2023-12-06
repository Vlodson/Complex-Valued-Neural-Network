from typing import Dict, List, Tuple
import numpy as np
import numpy.typing as npt


type ComplexMatrix = npt.NDArray[np.complex_]
type NamedObjectDict[T] = Dict[str, T]
type CategoricalLabels = npt.NDArray[np.int_]
type MiniBatches = List[Tuple[ComplexMatrix, CategoricalLabels]]
