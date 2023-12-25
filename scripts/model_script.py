from matplotlib import pyplot as plt

from layers.layer import Layer
from model.model import Model
from optimizers.adam import ADAM
from utils.loss_visualization import plot_complex_loss
from utils.synthetic_data import make_linear_multiclass_dataset


def main():
    x, y = make_linear_multiclass_dataset()
    x = x[:, 0] + x[:, 1] * 1j
    x = x.reshape(-1, 1)

    plt.scatter(x.real, x.imag, c=y)
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
        x=x_train, y=y_train, batch_size=64, epochs=1024, x_val=x_val, y_val=y_val
    )

    plot_complex_loss(model.history.history["train_loss"])
    plot_complex_loss(model.history.history["val_loss"])


if __name__ == "__main__":
    main()
