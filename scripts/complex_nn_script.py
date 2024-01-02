import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import wrapped_numpy as wnp
from losses.complex_mse_loss import ComplexMSE
from utils.synthetic_data import (
    make_linear_multiclass_dataset,
)
from utils.label_utils import cat_to_unit_vector
from utils.prediction_utils import vec_to_cat


def linear(x):
    return wnp.div(x, wnp.absolute(x))


def dlinear(x):
    return wnp.div(1, wnp.absolute(x))


def ctanh(x):
    return wnp.div(wnp.sub(wnp.exp(x), wnp.exp(-x)), wnp.add(wnp.exp(x), wnp.exp(-x)))


def dctanh(x):
    return wnp.sub(1, wnp.pwr(ctanh(x), 2))


def main():
    x, y = make_linear_multiclass_dataset()
    x = wnp.add(x[:, 0], x[:, 1] * 1j)
    x = x.reshape(-1, 1)

    plt.scatter(wnp.real(x), wnp.imag(x), c=y)
    plt.show()

    w1 = wnp.add(wnp.uniform(-1, 1, (1, 3)), wnp.uniform(-1, 1, (1, 3)) * 1.0j)
    b1 = wnp.add(wnp.uniform(-1, 1, (1, 3)), wnp.uniform(-1, 1, (1, 3)) * 1.0j)

    w2 = wnp.add(wnp.uniform(-1, 1, (3, 1)), wnp.uniform(-1, 1, (3, 1)) * 1.0j)
    b2 = wnp.add(wnp.uniform(-1, 1, (1, 1)), wnp.uniform(-1, 1, (1, 1)) * 1.0j)

    loss = ComplexMSE()
    losses = []
    n = 1e-1

    for _ in tqdm(range(1000)):
        t1 = wnp.add(wnp.dot(x, w1), b1)
        a1 = linear(t1)

        t2 = wnp.add(wnp.dot(a1, w2), b2)
        yh = linear(t2)

        losses.append(loss.calculate_loss(y, yh))
        dyh = loss.loss_gradient(y, yh)

        dt2 = wnp.mul(dlinear(t2), dyh)
        dw2 = wnp.dot(wnp.transpose(a1), dt2)
        db2 = np.sum(dt2, axis=0)
        da1 = wnp.dot(dt2, wnp.transpose(w2))

        dt1 = wnp.mul(dlinear(t1), da1)
        dw1 = wnp.dot(wnp.transpose(x), dt1)
        db1 = np.sum(dt1, axis=0)

        w2 = wnp.sub(w2, wnp.mul(n, dw2))
        b2 = wnp.sub(b2, wnp.mul(n, db2))

        w1 = wnp.sub(w1, wnp.mul(n, dw1))
        b1 = wnp.sub(b1, wnp.mul(n, db1))

    losses = np.array(losses)
    plt.scatter(wnp.real(losses), wnp.imag(losses), c=np.arange(losses.shape[0]))
    plt.colorbar(label="Epochs")
    plt.show()

    print(f"{cat_to_unit_vector(y[:5])}\n{yh[:5]}")

    preds = vec_to_cat(yh, 3)
    print(wnp.div(np.sum(preds == y), y.shape[0]))


if __name__ == "__main__":
    main()
