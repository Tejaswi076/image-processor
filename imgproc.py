"""Pure Python image processor (no NumPy / OpenCV / Pillow).

A grayscale image is a list of rows; each row is a list of ints 0-255.
An RGB image is a list of rows; each pixel is an [r, g, b] list.
Every operation returns a NEW image and never modifies its input.
"""


# ---------------------------------------------------------------- helpers
def clamp(v):
    """Force a value into the valid 0-255 range and make it an int."""
    return max(0, min(255, int(round(v))))


def copy_image(img):
    """Deep copy of a 2D grid."""
    return [row[:] for row in img]


def print_image(img, title=""):
    """Print a grayscale image as an aligned grid."""
    if title:
        print(title)
    for row in img:
        print("  " + " ".join(f"{v:3d}" for v in row))
    print()


# ------------------------------------------------- pixel-wise operations
def invert(img):
    """Photographic negative: each pixel p becomes 255 - p."""
    return [[255 - p for p in row] for row in img]


def adjust_brightness(img, delta):
    """Add delta to every pixel, clamping the result to 0-255."""
    return [[clamp(p + delta) for p in row] for row in img]


def threshold(img, cutoff):
    """Binary image: 255 where pixel >= cutoff, otherwise 0."""
    return [[255 if p >= cutoff else 0 for p in row] for row in img]


# ------------------------------------------------ geometric operations
def flip_horizontal(img):
    """Mirror left <-> right."""
    return [row[::-1] for row in img]


def flip_vertical(img):
    """Mirror top <-> bottom."""
    return [row[:] for row in img[::-1]]


def rotate_90_cw(img):
    """Rotate 90 degrees clockwise. Output has shape (width x height).

    New row r is old column r read from bottom to top.
    """
    h, w = len(img), len(img[0])
    return [[img[h - 1 - c][r] for c in range(h)] for r in range(w)]


def crop(img, top, left, height, width):
    """Return the height x width window whose top-left corner is (top, left)."""
    h, w = len(img), len(img[0])
    if top < 0 or left < 0 or top + height > h or left + width > w:
        raise ValueError("crop window falls outside the image")
    return [row[left:left + width] for row in img[top:top + height]]


# ------------------------------------------------ neighborhood operations
def get_pixel_clamped(img, r, c):
    """Read pixel (r, c); coordinates outside the image snap to the nearest
    edge pixel. This is the 'replicate' border policy."""
    h, w = len(img), len(img[0])
    r = max(0, min(h - 1, r))
    c = max(0, min(w - 1, c))
    return img[r][c]


def convolve(img, kernel, divisor=1, absolute=False):
    """Slide an odd-sized square kernel over the image.

    Borders: replicate (see get_pixel_clamped). Output keeps the input size.
    This computes correlation (kernel is not flipped); for the symmetric
    blur and the Sobel kernels below this only changes the sign of edges,
    which we remove with absolute=True.
    """
    k = len(kernel)
    off = k // 2
    h, w = len(img), len(img[0])
    out = []
    for r in range(h):
        new_row = []
        for c in range(w):
            total = 0
            for kr in range(k):
                for kc in range(k):
                    total += kernel[kr][kc] * get_pixel_clamped(img, r + kr - off, c + kc - off)
            total /= divisor
            new_row.append(clamp(abs(total) if absolute else total))
        out.append(new_row)
    return out


def box_blur(img):
    """3x3 average blur: each pixel becomes the mean of its neighborhood."""
    return convolve(img, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], divisor=9)


SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # responds to vertical edges
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]  # responds to horizontal edges


def sobel_edges(img):
    """Edge strength = |Gx| + |Gy|, clamped to 255 (a cheap magnitude)."""
    gx = convolve(img, SOBEL_X, absolute=True)
    gy = convolve(img, SOBEL_Y, absolute=True)
    return [[clamp(a + b) for a, b in zip(rx, ry)] for rx, ry in zip(gx, gy)]


# ------------------------------------------------------- bonus: RGB / PPM
def read_ppm(path):
    """Load an ASCII PPM (P3) file into a grid of [r, g, b] pixels."""
    with open(path) as f:
        tokens = []
        for line in f:
            tokens.extend(line.split("#")[0].split())  # strip comments
    if tokens[0] != "P3":
        raise ValueError("only ASCII P3 PPM files are supported")
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    vals = [int(t) * 255 // maxval for t in tokens[4:4 + w * h * 3]]
    return [[vals[(r * w + c) * 3:(r * w + c) * 3 + 3] for c in range(w)] for r in range(h)]


def write_ppm(img, path):
    """Save a grid of [r, g, b] pixels as an ASCII PPM (P3) file."""
    h, w = len(img), len(img[0])
    with open(path, "w") as f:
        f.write(f"P3\n{w} {h}\n255\n")
        for row in img:
            f.write(" ".join(f"{v}" for px in row for v in px) + "\n")


def to_grayscale(rgb):
    """Weighted luminance: 0.299 R + 0.587 G + 0.114 B."""
    return [[clamp(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in row] for row in rgb]


# ------------------------------------------------------------------ demo
def show(name, before, after):
    print("=" * 44)
    print(name)
    print("=" * 44)
    print_image(before, "BEFORE:")
    print_image(after, "AFTER:")


def main():
    # 5x5 test image: dark left, bright right, one mid-gray spot in row 1
    # (the spot makes flips and crops easy to verify by eye).
    img = [
        [ 10,  10,  10, 200, 200],
        [ 10,  10, 120, 200, 200],
        [ 10,  10,  10, 200, 200],
        [ 10,  10,  10, 200, 200],
        [ 10,  10,  10, 200, 200],
    ]
    snapshot = copy_image(img)

    show("invert", img, invert(img))
    show("adjust_brightness (+60)", img, adjust_brightness(img, 60))
    show("adjust_brightness (-60, clamps at 0)", img, adjust_brightness(img, -60))
    show("threshold (cutoff 100)", img, threshold(img, 100))

    show("flip_horizontal", img, flip_horizontal(img))
    show("flip_vertical", img, flip_vertical(img))

    # Non-square image so rotation visibly changes the shape (2 rows x 3 cols).
    small = [[1, 2, 3],
             [4, 5, 6]]
    show("rotate_90_cw (2x3 becomes 3x2)", small, rotate_90_cw(small))
    show("crop (top=1, left=2, height=3, width=3)", img, crop(img, 1, 2, 3, 3))

    show("box_blur (3x3, replicate borders)", img, box_blur(img))
    show("sobel_edges (replicate borders)", img, sobel_edges(img))

    # Bonus: RGB via PPM round trip.
    rgb = [[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
           [[255, 255, 0], [255, 255, 255], [0, 0, 0]]]
    write_ppm(rgb, "demo.ppm")
    loaded = read_ppm("demo.ppm")
    print("=" * 44)
    print("bonus: PPM round trip + to_grayscale")
    print("=" * 44)
    print("RGB written then read back identical:", loaded == rgb)
    print_image(to_grayscale(loaded), "Grayscale of the 2x3 RGB image:")

    print("Original image unchanged by all operations:", img == snapshot)


if __name__ == "__main__":
    main()
