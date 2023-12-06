from optimizers.optimizer import Optimizer
from optimizers.gradient_descent import GradientDescent
from optimizers.adam import ADAM
from custom_types import NamedObjectDict

OPTIMIZERS: NamedObjectDict[Optimizer] = {"gd": GradientDescent(), "adam": ADAM()}
