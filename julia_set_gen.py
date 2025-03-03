import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV to RGB"""
    h = (hsv[..., 0] * 6).astype(int)
    s = hsv[..., 1]
    v = hsv[..., 2]

    def hue_to_rgb(p, q, t):
        if t < 0: t += 1
        elif t > 1: t -= 1

        if t < 1/6:
            return v * (1 - s) + p * s
        elif t < 1/2:
            return v * (1 - s) + q * s
        elif t < 2/3:
            return v * (1 - s) + p * s
        else:
            return v

    r = hue_to_rgb(v, h % 6 == 0, s)
    g = hue_to_rgb(v, h % 6 == 1 or h % 6 == 5, s)
    b = hue_to_rgb(0, h % 2 == 0, s)

    return np.dstack((r, g, b)).astype(np.float32)


def render_julia_set(width: int, height: int) -> Image:
    """Render the Julia set"""
    x_range = (-1.5 + 0.3j, -0.8 + 0.156j)
    y_range = (-1.4, 1.6)

    c = complex(-0.7, 0.2)

    x = np.linspace(x_range[0].real, x_range[1].real, height)
    y = np.linspace(y_range[0], y_range[1], width)
    X, Y = np.meshgrid((x + 1j * (y - y_range[0])), y)
    Z = X + 1j * Y

    div_iter = np.zeros(Z.shape, dtype=int)
    mask = np.ones(Z.shape, dtype=bool)

    max_iter = 256
    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + c
        mask_new = np.abs(Z) <= 2
        div_iter[~mask & mask_new] = i
        mask = mask_new

    smooth = (div_iter + 1 - np.log(np.log2(np.abs(Z)))) / (smooth.max() - smooth.min())

    hsv = np.zeros((height, width, 3), dtype=float)
    hsv[..., 0] = (smooth + 1) % 1
    hsv[..., 1] = smooth * 0.8
    hsv[..., 2] = smooth ** 0.5

    rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
    img = Image.fromarray(rgb)

    return img


def postprocess_image(img: Image, alpha: float) -> Image:
    """Apply glow and contrast adjustment"""
    blur = img.filter(ImageFilter.GaussianBlur(radius=4))
    glow = Image.blend(img, blur, alpha)
    enhanced = ImageEnhance.Contrast(glow).enhance(1.5)

    return enhanced


def main():
    width, height = 800, 800
    img = render_julia_set(width, height)
    enhanced_img = postprocess_image(img, 0.2)

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(enhanced_img)
    plt.show()

if __name__ == "__main__":
    main()

