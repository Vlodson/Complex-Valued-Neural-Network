from custom_types import NamedObjectDict
from losses.loss import Loss
from losses.interval_distance_loss import IntervalDistance
from losses.argument_distance_loss import ArgumentDistance
from losses.complex_mse_loss import ComplexMSE

LOSSES: NamedObjectDict[Loss] = {
    "intDist": IntervalDistance(),
    "argDist": ArgumentDistance(),
    "mse": ComplexMSE(),
}
