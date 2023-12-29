import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from losses.complex64mse_loss import ComplexMSE
from utils.synthetic_data import (
    make_linear_multiclass_dataset,
)
from utils.label_utils import cat_to_unit_vector
from utils.prediction_utils import vec_to_cat


def linear(x):
    return x / np.absolute(x)


def dlinear(x):
    return 1 / np.absolute(x)


def ctanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def dctanh(x):
    return 1 - ctanh(x) ** 2


def main():
    x, y = make_linear_multiclass_dataset()
    x = x[:, 0] + x[:, 1] * 1j
    x = x.reshape(-1, 1)

    plt.scatter(x.real, x.imag, c=y)
    plt.show()

    w1 = np.random.uniform(-1, 1, (1, 3)) + np.random.uniform(-1, 1, (1, 3)) * 1.0j
    b1 = np.random.uniform(-1, 1, (1, 3)) + np.random.uniform(-1, 1, (1, 3)) * 1.0j

    w2 = np.random.uniform(-1, 1, (3, 1)) + np.random.uniform(-1, 1, (3, 1)) * 1.0j
    b2 = np.random.uniform(-1, 1, (1, 1)) + np.random.uniform(-1, 1, (1, 1)) * 1.0j

    loss = ComplexMSE()
    losses = []
    n = 1e-1

    for _ in tqdm(range(1000)):
        t1 = x @ w1 + b1
        a1 = linear(t1)

        t2 = a1 @ w2 + b2
        yh = linear(t2)

        losses.append(loss.calculate_loss(y, yh))
        dyh = loss.loss_gradient(y, yh)

        dt2 = dlinear(t2) * dyh
        dw2 = a1.T @ dt2
        db2 = np.sum(dt2, axis=0)
        da1 = dt2 @ w2.T

        dt1 = dlinear(t1) * da1
        dw1 = x.T @ dt1
        db1 = np.sum(dt1, axis=0)

        w2 -= n * dw2
        b2 -= n * db2

        w1 -= n * dw1
        b1 -= n * db1

    losses = np.array(losses)
    plt.scatter(losses.real, losses.imag, c=np.arange(losses.shape[0]))
    plt.colorbar(label="Epochs")
    plt.show()

    print(f"{cat_to_unit_vector(y[:5])}\n{yh[:5]}")

    preds = vec_to_cat(yh, 3)
    print(np.sum(preds == y) / y.shape[0])


if __name__ == "__main__":
    main()
