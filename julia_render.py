import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-1.76, 1.76)
y_range = (-1.76, 1.76)
c = complex(0.3, -0.51)
max_iter = 350

x = np.linspace(x_range[0], x_range[1], width)
y = np.linspace(y_range[0], y_range[1], height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

div_iter = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)
for i in range(max_iter):
    Z[mask] = Z[mask] ** 2 + c
    mask_new = np.abs(Z) <= 2
    div_iter[mask & ~mask_new] = i
    mask = mask_new

with np.errstate(divide='ignore', invalid='ignore'):
    smooth = div_iter + 1 - np.log(np.log2(np.abs(Z)))
    smooth = np.nan_to_num(smooth)
smooth_norm = (smooth - smooth.min()) / (smooth.max() - smooth.min())

# Bright rainbow palette
hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (smooth_norm + 0.3) % 1
hsv[..., 1] = 0.95 - 0.4 * smooth_norm
hsv[..., 2] = smooth_norm ** 0.4

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

def vertical_wave(im, amp=12, freq=0.09):
    arr = np.array(im)
    for j in range(arr.shape[1]):
        arr[:, j] = np.roll(arr[:, j], int(amp * np.sin(freq * j)))
    return Image.fromarray(arr)

img = vertical_wave(img, amp=18, freq=0.13)
img = ImageEnhance.Color(img).enhance(1.7)

output_path = 'julia_output.jpg'
img.save(output_path) 