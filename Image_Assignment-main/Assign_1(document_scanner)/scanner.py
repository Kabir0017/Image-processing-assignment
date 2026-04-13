# Name: Kabir Bidhuri
# Roll No: 2301010461
# Course: BTech CSE CORE
# Unit 1: Image acquisition, grayscale conversion, sampling awareness
# Assignment Title: Document Scanner Image Processing

from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib
import numpy as np


def pick_backend():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")


pick_backend()
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILES = [
    "printed_doc.png",
    "scannedpdf_image.webp",
    "photographed_doc_image.jpeg",
]


def build_parser():
    parser = ArgumentParser(description="Document scanning, sampling, and quantization demo.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a single document image.",
    )
    return parser


def resolve_images():
    if any((BASE_DIR / name).exists() for name in IMAGE_FILES):
        return [BASE_DIR / name for name in IMAGE_FILES if (BASE_DIR / name).exists()]
    raise FileNotFoundError(
        "No document image found in Assign_1(document_scanner). "
        "Add printed_doc.png, scannedpdf_image.webp, or photographed_doc_image.jpeg."
    )


def resolve_output_dir():
    out_dir = BASE_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def create_variants(gray):
    sample_512 = gray.copy()
    sample_256 = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    sample_256 = cv2.resize(sample_256, (512, 512), interpolation=cv2.INTER_NEAREST)
    sample_128 = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    sample_128 = cv2.resize(sample_128, (512, 512), interpolation=cv2.INTER_NEAREST)
    quant_256 = gray.copy()
    quant_16 = (gray // 16) * 16
    quant_4 = (gray // 64) * 64
    return sample_512, sample_256, sample_128, quant_256, quant_16, quant_4


def show_notes():
    print("Observations:")
    print("  - 512x512 keeps the text sharpest.")
    print("  - 256x256 starts softening edges a little.")
    print("  - 128x128 introduces obvious blockiness.")
    print("  - 16 and 4 level quantization show visible banding and detail loss.")


def process_one_image(image_path, output_dir):
    print(f"Loading image from {image_path}")
    color = read_image(image_path)
    color = cv2.resize(color, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    sample_512, sample_256, sample_128, quant_256, quant_16, quant_4 = create_variants(gray)

    show_notes()

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle(f"Document Scanner Quality Analysis - {image_path.stem}", fontsize=15, fontweight="bold")

    axes[0, 0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Resized")
    axes[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Grayscale")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(sample_512, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Sampling: 512x512")
    axes[1, 1].imshow(sample_256, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Sampling: 256x256")
    axes[1, 2].imshow(sample_128, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("Sampling: 128x128")

    axes[2, 0].imshow(quant_256, cmap="gray", vmin=0, vmax=255)
    axes[2, 0].set_title("Quantization: 256 Levels")
    axes[2, 1].imshow(quant_16, cmap="gray", vmin=0, vmax=255)
    axes[2, 1].set_title("Quantization: 16 Levels")
    axes[2, 2].imshow(quant_4, cmap="gray", vmin=0, vmax=255)
    axes[2, 2].set_title("Quantization: 4 Levels")

    for axis in axes.flat:
        if axis.has_data():
            axis.set_xticks([])
            axis.set_yticks([])

    plt.tight_layout()
    output_path = output_dir / f"{image_path.stem}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison figure to {output_path}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    print("=" * 55)
    print("Welcome to the Document Scanning and Processing System")
    print("=" * 55)

    output_dir = resolve_output_dir()
    if args.image:
        process_one_image(Path(args.image).expanduser(), output_dir)
    else:
        for image_path in resolve_images():
            process_one_image(image_path, output_dir)

    print(f"\nDone. Output folder: {output_dir}")


if __name__ == "__main__":
    main()
