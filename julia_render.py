import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib.colors import hsv_to_rgb

width, height = 1600, 1600
x_range = (-0.75, 0.75)
y_range = (-1.0, 1.0)
c = complex(-0.64, 0.44)
max_iter = 320

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

hsv = np.zeros((height, width, 3), dtype=float)
hsv[..., 0] = (0.7 * smooth_norm + 0.2) % 1
hsv[..., 1] = 0.95 - 0.1 * np.abs(np.sin(2 * np.pi * smooth_norm))
hsv[..., 2] = smooth_norm ** 0.2

rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
img = Image.fromarray(rgb)

# Sepia tone
def sepia(im):
    arr = np.array(im).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    tr = 0.393 * r + 0.769 * g + 0.189 * b
    tg = 0.349 * r + 0.686 * g + 0.168 * b
    tb = 0.272 * r + 0.534 * g + 0.131 * b
    arr[..., 0] = np.clip(tr, 0, 255)
    arr[..., 1] = np.clip(tg, 0, 255)
    arr[..., 2] = np.clip(tb, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

img = sepia(img)
img = img.filter(ImageFilter.GaussianBlur(radius=6))

def vertical_wave(im, amp=12, freq=0.09):
    arr = np.array(im)
    for j in range(arr.shape[1]):
        arr[:, j] = np.roll(arr[:, j], int(amp * np.sin(freq * j)))
    return Image.fromarray(arr)

img = vertical_wave(img, amp=18, freq=0.13)

output_path = 'julia_output.jpg'
img.save(output_path) 