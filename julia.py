
import numpy as np
import matplotlib.pyplot as plt

def compute_julia(width, height, c, max_iter):
    x, y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2, 2, height))
    z = x + 1j * y
    julia = np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        mask = (np.abs(z) < 2) & ((x >= -2.01) | (y <= 2.01))  # boundary handling
        z[mask] = z[mask]**2 + c
        julia[mask] = i
    
    return julia

def plot_julia(julia, cmap='viridis'):
    plt.imshow(julia, cmap=cmap)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    width, height = 500, 500
    c = complex(-0.8, 0.156)
    max_iter = 255
    
    julia = compute_julia(width, height, c, max_iter)
    plot_julia(julia)

