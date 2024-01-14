from layers.layer import Layer
from model.model import Model
from optimizers.adam import ADAM
from utils.synthetic_data import binary_classification


def main():
    features = 5

    x, y = binary_classification(features=features)

    test_cutoff = int(x.shape[0] * 0.8)
    x_test, y_test = x[test_cutoff:], y[test_cutoff:]

    x_train, y_train = x[:test_cutoff], y[:test_cutoff]

    val_cutoff = int(x_train.shape[0] * 0.8)
    x_val, y_val = x_train[val_cutoff:], y_train[val_cutoff:]
    x_train, y_train = x_train[:val_cutoff], y_train[:val_cutoff]

    model = Model(
        [
            Layer(neurons=3, activation="linsca", input_shape=x_train.shape[1]),
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
    print(model.test(x_test, y_test))


if __name__ == "__main__":
    main()
