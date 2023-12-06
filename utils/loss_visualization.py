from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_real_loss(loss: List[float]) -> None:
    plt.plot(loss)
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


def plot_complex_loss(loss: List[complex]) -> None:
    loss_arr = np.array(loss)

    plt.scatter(loss_arr.real, loss_arr.imag, c=np.arange(loss_arr.shape[0]))
    plt.colorbar(label="Epochs")
    plt.title("Loss")
    plt.xlabel("real")
    plt.ylabel("imaginary")
    plt.show()
