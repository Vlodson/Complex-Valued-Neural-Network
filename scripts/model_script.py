from matplotlib import pyplot as plt

import wrapped_numpy as wnp
from layers.layer import Layer
from model.model import Model
from optimizers.adam import ADAM
from utils.synthetic_data import make_linear_multiclass_dataset


def main():
    x, y = make_linear_multiclass_dataset()
    x = wnp.add(x[:, 0], x[:, 1] * 1j)
    x = x.reshape(-1, 1)

    plt.scatter(wnp.real(x), wnp.imag(x), c=y)
    plt.show()

    x_val, y_val = x[2700:], y[2700:]
    x_train, y_train = x[:2700], y[:2700]

    model = Model(
        [
            Layer(neurons=3, activation="linsca", input_shape=1),
            Layer(neurons=1, activation="linsca"),
        ]
    )
    model.build(
        optimizer=ADAM(1e-1),
        loss="mse",
        metrics=["accuracy", "recall", "precision", "f1"],
    )

    model.train(
        x=x_train, y=y_train, batch_size=64, epochs=512, x_val=x_val, y_val=y_val
    )

    model.history.plot_losses()
    print(model.test(x_train[:100], y_train[:100]))


if __name__ == "__main__":
    main()
