import numpy as np
import matplotlib.pyplot as plt

def julia_set(width, height, c, max_iter):
    x, y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2, 2, height))
    z = x + 1j * y
    julia = np.zeros(z.shape, dtype=int)
    for i in range(max_iter):
        mask = np.abs(z) < 2
        z[mask] = z[mask]**2 + c
        julia[mask] = i
    return julia

if __name__ == '__main__':
    width, height = 500, 500
    c = complex(-0.8, 0.156)
    max_iter = 255
    julia = julia_set(width, height, c, max_iter)
    plt.imshow(julia, cmap='hot')
    plt.axis('off')
    plt.show()
