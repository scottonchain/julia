import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.colors import hsv_to_rgb


def compute_julia(
    width=1600,
    height=1600,
    center=(-0.745, 0.186),
    scale=0.045,
    c=complex(-0.51, 0.55),
    max_iter=700,
    escape_radius=2.0,
):
    """
    Compute a Julia set with smooth escape-time coloring.
    """
    x_range = (center[0] - scale, center[0] + scale)
    y_range = (center[1] - scale, center[1] + scale)

    x = np.linspace(x_range[0], x_range[1], width, dtype=np.float64)
    y = np.linspace(y_range[0], y_range[1], height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    Z = X + 1j * Y

    escaped = np.zeros(Z.shape, dtype=bool)
    smooth = np.zeros(Z.shape, dtype=np.float64)

    escape_radius_sq = escape_radius * escape_radius

    for i in range(max_iter):
        active = ~escaped
        if not np.any(active):
            break

        Z[active] = Z[active] * Z[active] + c
        mag_sq = Z.real * Z.real + Z.imag * Z.imag

        just_escaped = active & (mag_sq > escape_radius_sq)
        if np.any(just_escaped):
            escaped[just_escaped] = True
            abs_z = np.sqrt(mag_sq[just_escaped])
            smooth[just_escaped] = i + 1 - np.log2(np.log(abs_z))

    if np.any(escaped):
        smin = smooth[escaped].min()
        # A few very late escapes should not dim the rest of the image.
        smax = np.percentile(smooth[escaped], 99.3)
        if smax > smin:
            smooth[escaped] = np.clip(
                (smooth[escaped] - smin) / (smax - smin), 0.0, 1.0
            )
        else:
            smooth[escaped] = 0.0

    smooth[~escaped] = 0.0
    return smooth, escaped


def colorize_julia(
    smooth,
    escaped,
    hue_offset=0.68,
    hue_scale=0.95,
    saturation=0.9,
    interior_value=0.015,
    gamma=0.8,
    palette=None,
):
    """
    Color smooth values with HSV or seven RGB palette anchors.
    """
    if palette is not None:
        tones = np.power(smooth, gamma)
        stops = np.array([0.0, 0.20, 0.40, 0.58, 0.76, 0.90, 1.0])
        colors = np.asarray(palette, dtype=np.float64)
        rgb = np.stack(
            [np.interp(tones, stops, colors[:, channel]) for channel in range(3)],
            axis=-1,
        )
        rgb[~escaped] = (5, 8, 16)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    h, w = smooth.shape
    hsv = np.zeros((h, w, 3), dtype=np.float64)

    hsv[..., 0] = (hue_offset + hue_scale * smooth) % 1.0
    hsv[..., 1] = saturation
    hsv[..., 2] = np.power(smooth, gamma)

    hsv[~escaped, 1] = 0.0
    hsv[~escaped, 2] = interior_value

    rgb = hsv_to_rgb(hsv)
    rgb8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb8, mode="RGB")


def pixel_sort_by_luminance(
    image,
    threshold=75,
    min_run_length=20,
    sort_descending=False,
):
    """
    Pixel-sort contiguous bright regions in each row using luminance.
    """
    arr = np.array(image)
    out = arr.copy()

    lum = (
        0.2126 * arr[..., 0]
        + 0.7152 * arr[..., 1]
        + 0.0722 * arr[..., 2]
    )

    for y in range(arr.shape[0]):
        row = out[y]
        row_lum = lum[y]
        bright = row_lum >= threshold

        start = None
        for x in range(len(bright) + 1):
            on = x < len(bright) and bright[x]
            if on and start is None:
                start = x
            elif not on and start is not None:
                end = x
                if end - start >= min_run_length:
                    segment = row[start:end]
                    seg_lum = row_lum[start:end]
                    order = np.argsort(seg_lum)
                    if sort_descending:
                        order = order[::-1]
                    row[start:end] = segment[order]
                start = None

    return Image.fromarray(out, mode="RGB")


def enhance_image(
    image,
    color=1.8,
    contrast=1.25,
    brightness=1.1,
):
    """
    Apply gentle post-processing.
    """
    image = ImageEnhance.Color(image).enhance(color)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    return image


def main():
    output_size = 1600
    supersample = 2
    smooth, escaped = compute_julia(
        width=output_size * supersample,
        height=output_size * supersample,
        center=(-0.105, 0.755),
        scale=0.235,
        c=complex(-0.513, 0.553),
        max_iter=500,
        escape_radius=16.0,
    )

    img = colorize_julia(
        smooth,
        escaped,
        gamma=0.57,
        # Lapis, sea glass, and antique-gold highlights.
        palette=[
            (5, 8, 20), (15, 31, 58), (32, 68, 97),
            (67, 127, 143), (145, 184, 181), (205, 177, 123),
            (244, 224, 181),
        ],
    )
    img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

    sorted_img = pixel_sort_by_luminance(
        img,
        threshold=175,
        min_run_length=60,
        sort_descending=False,
    )
    img = Image.blend(img, sorted_img, 0.08)

    img = enhance_image(
        img,
        color=1.03,
        contrast=1.02,
        brightness=1.02,
    )

    output_path = "julia_output.jpg"
    img.save(output_path, quality=95, subsampling=0, optimize=True)
    print(f"Saved {output_path}")

    output_path = "julia_output.png"
    img.save(output_path, optimize=True)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
