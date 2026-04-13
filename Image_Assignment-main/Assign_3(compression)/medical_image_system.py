# Name: Kabir Bidhuri
# Roll No: 2301010461
# Course: BTech CSE CORE
# Unit 3: Segmentation, thresholding, morphology
# Assignment Title: Medical Image Compression, Segmentation, and Morphological Processing
# Date: 2026-04-09

from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib
import numpy as np


def choose_backend():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")


choose_backend()
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES = {
    "xray": BASE_DIR / "xray_img.jpg",
    "mri": BASE_DIR / "MRI_img.jpg",
    "ct": BASE_DIR / "CT SCAN.jpeg",
}


def build_parser():
    parser = ArgumentParser(description="Medical image compression and segmentation demo.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to one medical image. If omitted, all default images are processed.",
    )
    return parser


def out_dir():
    folder = BASE_DIR / "output"
    folder.mkdir(exist_ok=True)
    return folder


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def grayscale(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rle_encode(gray):
    flat = gray.flatten()
    if flat.size == 0:
        return []

    runs = []
    value = int(flat[0])
    count = 1

    for item in flat[1:]:
        item = int(item)
        if item == value:
            count += 1
        else:
            runs.append((value, count))
            value = item
            count = 1

    runs.append((value, count))
    return runs


def rle_stats(gray, runs):
    original_bytes = gray.size
    compressed_bytes = len(runs) * 2
    ratio = original_bytes / compressed_bytes if compressed_bytes else float("inf")
    savings = (1.0 - compressed_bytes / original_bytes) * 100 if original_bytes else 0.0
    return original_bytes, compressed_bytes, ratio, savings


def threshold_global(gray):
    threshold_value = int(np.mean(gray))
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return threshold_value, binary


def threshold_otsu(gray):
    threshold_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(threshold_value), binary


def morph(binary):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1), cv2.erode(binary, kernel, iterations=1)


def roi_summary(binary):
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    items = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        items.append((area, x, y, w, h))
    items.sort(reverse=True)
    return items[:5]


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def draw_task1(gray, stem, image_name, runs, stats, folder):
    original_bytes, compressed_bytes, ratio, savings = stats

    section(f"Task 1: Image Compression [{stem}]")
    print(f"Loaded grayscale medical image: {image_name}")
    print(f"RLE runs: {len(runs)}")
    print(f"Original size: {original_bytes} bytes")
    print(f"Compressed size estimate: {compressed_bytes} bytes")
    print(f"Compression ratio: {ratio:.2f}")
    print(f"Storage savings: {savings:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Task 1 - Compression [{stem}]", fontsize=15, fontweight="bold")
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Grayscale Medical Image")
    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.95,
        "\n".join(
            [
                f"RLE runs: {len(runs)}",
                f"Original bytes: {original_bytes}",
                f"Compressed bytes: {compressed_bytes}",
                f"Compression ratio: {ratio:.2f}",
                f"Storage savings: {savings:.2f}%",
            ]
        ),
        va="top",
        fontsize=12,
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.tight_layout()
    save_figure(fig, folder / f"{stem}_task1_compression.png")
    plt.close(fig)


def draw_task2(gray, stem, g_value, g_binary, o_value, o_binary, g_rois, o_rois, folder):
    section(f"Task 2: Image Segmentation [{stem}]")
    print(f"Global threshold value: {g_value}")
    print(f"Otsu threshold value: {o_value}")
    print(f"Global segmentation ROIs: {len(g_rois)}")
    print(f"Otsu segmentation ROIs: {len(o_rois)}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Task 2 - Segmentation [{stem}]", fontsize=15, fontweight="bold")
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Grayscale")
    axes[0, 1].imshow(g_binary, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Global Threshold ({g_value})")
    axes[0, 2].imshow(o_binary, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title(f"Otsu Threshold ({o_value})")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    def fmt(items):
        if not items:
            return "None"
        return "\n".join([f"Area={a}, x={x}, y={y}, w={w}, h={h}" for a, x, y, w, h in items[:3]])

    axes[1, 0].text(
        0.0,
        0.9,
        f"Top ROIs - Global:\n{fmt(g_rois)}\n\nTop ROIs - Otsu:\n{fmt(o_rois)}",
        va="top",
        fontsize=10,
    )
    for ax in axes.flat:
        if ax.has_data():
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    save_figure(fig, folder / f"{stem}_task2_segmentation.png")
    plt.close(fig)


def draw_task3(stem, g_binary, g_dilated, g_eroded, o_binary, o_dilated, o_eroded, folder):
    section(f"Task 3: Morphological Processing [{stem}]")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Task 3 - Morphology [{stem}]", fontsize=15, fontweight="bold")
    axes[0, 0].imshow(g_binary, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Global Binary")
    axes[0, 1].imshow(g_dilated, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Global Dilation")
    axes[0, 2].imshow(g_eroded, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Global Erosion")
    axes[1, 0].imshow(o_binary, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Otsu Binary")
    axes[1, 1].imshow(o_dilated, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Otsu Dilation")
    axes[1, 2].imshow(o_eroded, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("Otsu Erosion")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    save_figure(fig, folder / f"{stem}_task3_morphology.png")
    plt.close(fig)


def analyze(stem, g_rois, o_rois):
    section(f"Task 4: Analysis and Interpretation [{stem}]")
    print("Segmentation comparison:")
    print(f"  Global threshold extracted {len(g_rois)} major connected regions.")
    print(f"  Otsu threshold extracted {len(o_rois)} major connected regions.")
    print("Clinical relevance:")
    print("  - Thresholding separates anatomical structures from background intensity.")
    print("  - Otsu works well when the histogram has two clear groups.")
    print("  - Dilation reconnects broken areas; erosion removes small noise blobs.")
    print("  - The better result is the one that keeps the target region clear and compact.")


def process_one(path, folder):
    image = load_image(path)
    gray = grayscale(image)
    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)

    runs = rle_encode(gray)
    stats = rle_stats(gray, runs)

    g_value, g_binary = threshold_global(gray)
    o_value, o_binary = threshold_otsu(gray)

    g_dilated, g_eroded = morph(g_binary)
    o_dilated, o_eroded = morph(o_binary)

    g_rois = roi_summary(g_binary)
    o_rois = roi_summary(o_binary)

    stem = path.stem.replace(" ", "_")
    draw_task1(gray, stem, path.name, runs, stats, folder)
    draw_task2(gray, stem, g_value, g_binary, o_value, o_binary, g_rois, o_rois, folder)
    draw_task3(stem, g_binary, g_dilated, g_eroded, o_binary, o_dilated, o_eroded, folder)
    analyze(stem, g_rois, o_rois)


def main():
    args = build_parser().parse_args()
    print("=" * 60)
    print("Welcome to the Medical Image Processing System")
    print("This script covers compression, segmentation, and morphology.")
    print("=" * 60)

    folder = out_dir()
    if args.image:
        process_one(Path(args.image).expanduser(), folder)
    else:
        for path in DEFAULT_IMAGES.values():
            process_one(path, folder)

    print(f"\nSample run complete.\nOutputs saved in: {folder}")


if __name__ == "__main__":
    main()
