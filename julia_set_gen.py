import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

__doc__ = "Julia Set Visualization"

WIDTH, HEIGHT = 800, 800
X_RANGE = (-0.5 + 1j * 0.2, -0.4 + 1j * 0)
Y_RANGE = (-1.3, 1.3)
C = complex(-0.35, 0.25)

MAX_ITER = 300

ARTISTIC_ALPHA = 2.5


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV to RGB"""
    h, s, v = hsv.T
    r = v * (1 - s)
    g = v * ((h % 6 / 6) + 0.5)
    b = v * min(s, 1)
    return (r, g, b).reshape(-1, 3)


def generate_julia_set(x_min: float, x_max: float, y_min: float, y_max: float,
                        c: complex, max_iter: int) -> tuple:
    """Generate the Julia set"""
    x = np.linspace(x_min, x_max, WIDTH)
    y = np.linspace(y_min, y_max, HEIGHT)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    div_iter = np.zeros(Z.shape, dtype=int)
    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + c
        mask_new = np.abs(Z) <= 2
        div_iter[~mask & mask_new] = i
        mask = mask_new

    return X, Y, Z, div_iter


def visualize_julia_set(x: np.ndarray, y: np.ndarray, z: complex,
                         max_iter: int) -> tuple:
    """Visualize the Julia set"""
    smooth = (div_iter + 1 - np.log(np.log2(np.abs(z)))) / (
        div_iter.max() - div_iter.min())
    smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

    hsv = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
    hsv[..., 0] = smooth_norm / 3
    hsv[..., 1] = 0.8 + 0.4 * smooth_norm
    hsv[..., 2] = smooth_norm ** 0.3

    rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
    img = Image.fromarray(rgb)

    blur = img.filter(ImageFilter.GaussianBlur(radius=7))
    glow = Image.blend(img, blur, alpha=ARTISTIC_ALPHA)
    enhanced_img = ImageEnhance.Brightness(glow).enhance(1.5)

    return enhanced_img


def main():
    x_min, x_max = X_RANGE[0].real, X_RANGE[1].real
    y_min, y_max = Y_RANGE[0], Y_RANGE[1]
    c = complex(-0.35, 0.25)
    max_iter = MAX_ITER

    try:
        X, Y, Z, div_iter = generate_julia_set(x_min, x_max, y_min, y_max,
                                                 c, max_iter)

        enhanced_img = visualize_julia_set(X, Y, z=c, max_iter=max_iter)

        # Save the image
        enhanced_img.save("julia.png")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
import os; print("Current directory:", os.getcwd())

