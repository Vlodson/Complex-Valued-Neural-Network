from typing import List

import matplotlib.pyplot as plt
import numpy as np
import wrapped_numpy as wnp


def plot_real_loss(loss: List[float]) -> None:
    plt.plot(loss)
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


def plot_complex_loss(loss: List[complex]) -> None:
    loss_arr = np.array(loss)

    plt.scatter(wnp.real(loss_arr), wnp.imag(loss_arr), c=np.arange(loss_arr.shape[0]))
    plt.colorbar(label="Epochs")
    plt.title("Loss")
    plt.xlabel("real")
    plt.ylabel("imaginary")
    plt.show()


def plot_real_metrics(metrics: List[float], name: str) -> None:
    plt.plot(metrics)
    plt.title(f"Metric: {name}")
    plt.xlabel("epochs")
    plt.ylabel(name)
    plt.show()
