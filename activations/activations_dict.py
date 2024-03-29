from activations.activation import Activation
from activations.complex_tanh import ComplexTanH
from activations.linear_scaled import LinearScaled
from activations.unit import Unit
from custom_types import NamedObjectDict

ACTIVATIONS: NamedObjectDict[Activation] = {
    "ctanh": ComplexTanH(),
    "unit": Unit(),
    "linsca": LinearScaled(),
}
