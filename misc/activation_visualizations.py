from typing import Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt


def make_unit_square() -> Tuple[np.ndarray[float], np.ndarray[float]]:
    return np.meshgrid(np.linspace(-1, 1, 250), np.linspace(-1, 1, 250))


def make_unit_circle() -> Tuple[np.ndarray[float], np.ndarray[float]]:
    return (
        np.cos(np.linspace(0, 2 * np.pi, 1000)),
        np.sin(np.linspace(0, 2 * np.pi, 1000)),
    )


def stylize_plot(ax: plt.Axes, title: str) -> plt.Axes:
    ax.set_title(title)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")

    ax.spines["left"].set_position("zero")

    ax.spines["right"].set_color("none")
    ax.yaxis.tick_left()

    ax.spines["bottom"].set_position("zero")

    ax.spines["top"].set_color("none")
    ax.xaxis.tick_bottom()

    ax.grid(visible=True, which="major", axis="both")
    return ax


def color_area(
    ax: plt.Axes,
    re: np.ndarray,
    im: np.ndarray,
    color: Literal["r", "g", "b", "y"] = "b",
    alpha: float = 0.3,
) -> plt.Axes:
    ax.fill(re, im, color=color, alpha=alpha)

    return ax


def real_relu(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    return np.where(z.real > 0, z.real, 0) + z.imag * 1.0j


def imag_relu(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    return z.real + np.where(z.imag > 0, z.imag, 0) * 1.0j


def strong_relu(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    return np.where(z.real > 0, z.real, 0) + np.where(z.imag > 0, z.imag, 0) * 1.0j


def strong_elu(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    a = 1e-0
    return np.where((z.real > 0) & (z.imag > 0), z, a * (np.exp(z) - 1))


def strong_selu(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    l, a = 1.0507, 1.67326  # textbook lambda and alpha selu values
    return np.where((z.real > 0) & (z.imag > 0), l * z, l * a * (np.exp(z) - 1))


def tanh(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def sigmoid(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    return 1 / (1 + np.exp(-z))


def swish(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    return z / (1 + np.exp(-z))


def gaussian(z: np.ndarray[np.complex64]) -> np.ndarray[np.complex64]:
    return np.exp(-(z**2))


def caaf(z: np.ndarray[np.complex64]) -> float:
    # complex argument activation function
    return np.angle(z)


def single_ccl(y_hat: float, angle_range: Tuple[float, float]) -> float:
    return (
        (y_hat - angle_range[0]) ** 2
        if y_hat < angle_range[0]
        else (y_hat - angle_range[1]) ** 2
        if y_hat > angle_range[1]
        else 0
    )


def label_to_range(y: int, ranges: np.ndarray) -> Tuple[float, float]:
    return ranges[y], ranges[y + 1]


def ccl(y: np.ndarray[int], y_hat: np.ndarray[float], classes: int) -> float:
    """
    y_hat - array of arguments of last layer
    y - zones to which the label belongs iside the unit sphere,
        counting counter-clockwise from 0 to 2pi
    classes - number of unique classes
    """
    ranges = np.linspace(0, 2 * np.pi, classes)


def main():
    r = 1
    # re, im = make_unit_square()
    re, im = make_unit_circle()
    re, im = r * re, r * im

    fz = swish(re + im * 1.0j)

    _, ax = plt.subplots(1, 2)

    ax[0] = color_area(ax[0], re, im)
    ax[0] = stylize_plot(ax[0], "Domain")

    ax[1] = color_area(ax[1], fz.real, fz.imag, "r")
    ax[1] = stylize_plot(ax[1], "Reach")

    plt.plot()
    plt.show()


if __name__ == "__main__":
    main()
