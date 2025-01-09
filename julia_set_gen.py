import numpy as np
from matplotlib import pyplot as plt

# Artistic Julia set parameters
image_size = 800
x_range = (-0.75 + 1j * 0.25, -0.5 + 1j * 0.2)  # Change the center of the julia set
y_range = (-1.8, 1.8)
c = complex(-0.7, 0.3)  # Tweak this for different shapes
max_iterations = 300

# Generate grid of complex points
x = np.linspace(x_range[0].real, x_range[1].real, image_size)
y = np.linspace(y_range[0], y_range[1], image_size)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Initialize iteration counts and mask
div_iterations = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

# Iterate and record divergence
for i in range(max_iterations):
    Z[mask] = Z[mask]**2 + c
    mask_new = np.abs(Z) <= 2
    div_iterations[~mask & mask_new] = i
    mask = mask_new

# Smooth coloring
smooth = (div_iterations - smooth.min()) / (smooth.max() - smooth.min())

# Build HSV image
hsv = np.zeros((image_size, image_size, 3), dtype=float)
hsv[..., 0] = (smooth + 0.6) % 1  # Hue
hsv[..., 1] = 0.8 + 0.2 * smooth  # Saturation
hsv[..., 2] = smooth ** 0.5  # Value

# Convert to RGB and display
rgb = hsv_to_rgb(hsv) * 255
img = rgb.astype(np.uint8)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img)
plt.show()

