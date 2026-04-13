# Name: Kabir Bidhuri
# Roll No: 2301010461
# Course: BTech CSE CORE
# Unit 2: Noise modeling, filtering, image enhancement
# Assignment Title: Image Restoration Using Spatial Filters
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
    "street": BASE_DIR / "street.jpg",
    "corridor": BASE_DIR / "corridor.jpg",
    "parking_lot": BASE_DIR / "parking_lot.jpeg",
}


def build_parser():
    parser = ArgumentParser(description="Noise modeling and restoration for surveillance-style images.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to one image. If skipped, all default images are processed.",
    )
    return parser


def output_dir():
    target = BASE_DIR / "outputs"
    target.mkdir(exist_ok=True)
    return target


def load_gray(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {path}")
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), image


def gaussian_noise(gray, sigma=20.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, gray.shape).astype(np.float32)
    result = gray.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def salt_pepper_noise(gray, amount=0.08, seed=42):
    rng = np.random.default_rng(seed)
    result = gray.copy()
    total = gray.size

    salt = int(np.ceil(total * amount * 0.5))
    pepper = int(np.ceil(total * amount * 0.5))

    salt_coords = tuple(rng.integers(0, dim, salt) for dim in gray.shape)
    pepper_coords = tuple(rng.integers(0, dim, pepper) for dim in gray.shape)

    result[salt_coords] = 255
    result[pepper_coords] = 0
    return result


def smooth_mean(img):
    return cv2.blur(img, (5, 5))


def smooth_median(img):
    return cv2.medianBlur(img, 5)


def smooth_gaussian(img):
    return cv2.GaussianBlur(img, (5, 5), 1.2)


def mse(ref, test):
    diff = ref.astype(np.float32) - test.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr(ref, test):
    error = mse(ref, test)
    return float("inf") if error == 0 else float(10 * np.log10((255.0 * 255.0) / error))


def top_filter(ref, filtered):
    ranked = sorted(
        ((mse(ref, img), -psnr(ref, img), name) for name, img in filtered.items())
    )
    return ranked[0][2]


def save_plot(fig, out_path):
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def title_block(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def report_metrics(ref, filtered):
    for name, img in filtered.items():
        print(f"  {name:<15} MSE = {mse(ref, img):.2f}, PSNR = {psnr(ref, img):.2f} dB")


def process_one(path, out_dir):
    gray, color = load_gray(path)
    g_noise = gaussian_noise(gray)
    sp_noise = salt_pepper_noise(gray)

    g_restored = {
        "Mean Filter": smooth_mean(g_noise),
        "Median Filter": smooth_median(g_noise),
        "Gaussian Filter": smooth_gaussian(g_noise),
    }
    sp_restored = {
        "Mean Filter": smooth_mean(sp_noise),
        "Median Filter": smooth_median(sp_noise),
        "Gaussian Filter": smooth_gaussian(sp_noise),
    }

    stem = path.stem.replace(" ", "_")
    print(f"Loading surveillance-style image from: {path}")

    title_block(f"Task 1: Image Selection and Preprocessing [{stem}]")
    print("Loaded the image, resized it to 512x512, and converted it to grayscale.")

    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
    fig1.suptitle(f"Task 1 - Original Image and Grayscale [{stem}]", fontsize=15, fontweight="bold")
    ax1[0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    ax1[0].set_title("Original Image")
    ax1[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax1[1].set_title("Grayscale Image")
    for ax in ax1:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    save_plot(fig1, out_dir / f"{stem}_task1_original_gray.png")
    plt.close(fig1)

    title_block(f"Task 2: Noise Modeling [{stem}]")
    print("Generated Gaussian noise and salt-and-pepper noise.")

    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 6))
    fig2.suptitle(f"Task 2 - Noise Modeling [{stem}]", fontsize=15, fontweight="bold")
    ax2[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax2[0].set_title("Original Grayscale")
    ax2[1].imshow(g_noise, cmap="gray", vmin=0, vmax=255)
    ax2[1].set_title("Gaussian Noise")
    ax2[2].imshow(sp_noise, cmap="gray", vmin=0, vmax=255)
    ax2[2].set_title("Salt-and-Pepper Noise")
    for ax in ax2:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    save_plot(fig2, out_dir / f"{stem}_task2_noisy_images.png")
    plt.close(fig2)

    title_block(f"Task 3: Image Restoration Techniques [{stem}]")
    print("Applied mean, median, and Gaussian filters to both noisy images.")

    fig3, ax3 = plt.subplots(2, 4, figsize=(18, 9))
    fig3.suptitle(f"Task 3 - Restoration Results [{stem}]", fontsize=15, fontweight="bold")
    ax3[0, 0].imshow(g_noise, cmap="gray", vmin=0, vmax=255)
    ax3[0, 0].set_title("Gaussian Noise")
    ax3[0, 1].imshow(g_restored["Mean Filter"], cmap="gray", vmin=0, vmax=255)
    ax3[0, 1].set_title("Mean Filter")
    ax3[0, 2].imshow(g_restored["Median Filter"], cmap="gray", vmin=0, vmax=255)
    ax3[0, 2].set_title("Median Filter")
    ax3[0, 3].imshow(g_restored["Gaussian Filter"], cmap="gray", vmin=0, vmax=255)
    ax3[0, 3].set_title("Gaussian Filter")

    ax3[1, 0].imshow(sp_noise, cmap="gray", vmin=0, vmax=255)
    ax3[1, 0].set_title("Salt-and-Pepper Noise")
    ax3[1, 1].imshow(sp_restored["Mean Filter"], cmap="gray", vmin=0, vmax=255)
    ax3[1, 1].set_title("Mean Filter")
    ax3[1, 2].imshow(sp_restored["Median Filter"], cmap="gray", vmin=0, vmax=255)
    ax3[1, 2].set_title("Median Filter")
    ax3[1, 3].imshow(sp_restored["Gaussian Filter"], cmap="gray", vmin=0, vmax=255)
    ax3[1, 3].set_title("Gaussian Filter")
    for ax in ax3.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    save_plot(fig3, out_dir / f"{stem}_task3_restored_images.png")
    plt.close(fig3)

    title_block(f"Task 4: Performance Evaluation [{stem}]")
    print("Gaussian Noise Restoration Metrics:")
    report_metrics(gray, g_restored)
    print("\nSalt-and-Pepper Noise Restoration Metrics:")
    report_metrics(gray, sp_restored)

    print("\nBest method by noise type:")
    print(f"  Gaussian noise: {top_filter(gray, g_restored)}")
    print(f"  Salt-and-pepper noise: {top_filter(gray, sp_restored)}")
    print("\nTheory-based justification:")
    print("  - Mean and Gaussian filters work well for additive Gaussian noise.")
    print("  - Median filtering usually handles salt-and-pepper noise better because it rejects outliers.")
    print("  - On these image types, median filtering keeps edges readable while cleaning impulse noise.")


def main():
    args = build_parser().parse_args()

    print("=" * 60)
    print("Welcome to the Image Restoration and Noise Analysis System")
    print("=" * 60)

    out_dir = output_dir()
    if args.image:
        process_one(Path(args.image).expanduser(), out_dir)
    else:
        for path in DEFAULT_IMAGES.values():
            process_one(path, out_dir)

    print(f"\nSample run complete.\nOutputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
