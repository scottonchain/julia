import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def render_julia_set(width: int, height: int) -> Image:
    """Render the Julia set"""
    
    x_range = (-1.5 + 0.3j, -0.8 + 0.156j)
    y_range = (-1.4, 1.6)

    c = complex(-0.7, 0.2)  # Adjust center of Julia set

    x = np.linspace(x_range[0].real, x_range[1].real, height)
    y = np.linspace(y_range[0], y_range[1], width)
    X, Y = np.meshgrid((x + 1j * (y - y_range[0])), y)
    Z = c + (X - x_range[0].real) * 1j + (Y - y_range[0])

    div_iter = np.zeros(Z.shape, dtype=int)
    mask = np.ones(Z.shape, dtype=bool)

    max_iter = 256
    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + c
        mask_new = np.abs(Z) <= 2
        div_iter[~mask & mask_new] = i
        mask &= ~mask_new

    return Image.fromarray((div_iter / max_iter * 255).astype(np.uint8))

def main():
    img = render_julia_set(800, 600)
    enhancer = ImageEnhance.Color(img)
    enhanced_img = enhancer.enhance(1.5)  # Adjust color scheme
    enhanced_img.show()

if __name__ == "__main__":
    main()

