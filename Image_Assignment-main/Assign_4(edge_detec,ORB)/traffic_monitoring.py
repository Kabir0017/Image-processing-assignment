# Name: Kabir Bidhuri
# Roll No: 2301010461
# Course: BTech CSE CORE
# Unit 4: Edge detection, object representation, feature extraction
# Assignment Title: Traffic Monitoring with Edge Detection and ORB
# Date: 2026-04-09

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
IMAGE_SET = {
    "road_intersection": BASE_DIR / "road_intersection.jpeg",
    "highway": BASE_DIR / "highway.jpeg",
    "pedestrian_crossing": BASE_DIR / "pedestrian crossings.jpeg",
}


def build_parser():
    parser = ArgumentParser(description="Traffic edge detection, contouring, and ORB feature demo.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to one traffic image. If skipped, all default traffic images are processed.",
    )
    return parser


def output_dir():
    folder = BASE_DIR / "outputs"
    folder.mkdir(exist_ok=True)
    return folder


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def sobel_edges(gray):
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(x, y)
    mag = np.uint8(np.clip(mag, 0, 255))
    return mag


def canny_edges(gray):
    return cv2.Canny(gray, 80, 160)


def contour_data(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    items = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        items.append((area, perimeter, x, y, w, h, contour))

    items.sort(key=lambda item: item[0], reverse=True)
    return items[:12], binary


def orb_features(gray):
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    marked = cv2.drawKeypoints(
        gray,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return keypoints, descriptors, marked


def save_plot(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def print_block(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def format_top_contours(contours):
    lines = []
    for idx, (area, perimeter, x, y, w, h, _) in enumerate(contours[:5], start=1):
        lines.append(
            f"{idx}. area={area:.1f}, perimeter={perimeter:.1f}, box=({x}, {y}, {w}, {h})"
        )
    return lines or ["No large contours found."]


def process_image(image_path, folder):
    print(f"Loading traffic image from: {image_path}")
    color = read_image(image_path)
    color = cv2.resize(color, (960, 540), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    sobel = sobel_edges(gray)
    canny = canny_edges(gray)

    contours, binary = contour_data(gray)
    contour_view = color.copy()
    for area, perimeter, x, y, w, h, contour in contours:
        cv2.drawContours(contour_view, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(contour_view, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            contour_view,
            f"A:{int(area)} P:{int(perimeter)}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    keypoints, descriptors, orb_view = orb_features(gray)
    stem = image_path.stem.replace(" ", "_")

    print_block(f"Task 1: Edge Detection [{stem}]")
    print("Applied Sobel and Canny edge detectors.")
    print(f"Sobel edge map shape: {sobel.shape}")
    print(f"Canny edge map shape: {canny.shape}")

    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle(f"Task 1 - Edge Detection [{stem}]", fontsize=15, fontweight="bold")
    axes1[0].imshow(gray, cmap="gray")
    axes1[0].set_title("Grayscale")
    axes1[1].imshow(sobel, cmap="gray")
    axes1[1].set_title("Sobel")
    axes1[2].imshow(canny, cmap="gray")
    axes1[2].set_title("Canny")
    for axis in axes1:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    save_plot(fig1, folder / f"{stem}_task1_edges.png")
    plt.close(fig1)

    print_block(f"Task 2: Object Representation [{stem}]")
    print("Detected contours and drew bounding boxes.")
    for line in format_top_contours(contours):
        print("  " + line)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f"Task 2 - Contours and Boxes [{stem}]", fontsize=15, fontweight="bold")
    axes2[0].imshow(binary, cmap="gray")
    axes2[0].set_title("Binary Image")
    axes2[1].imshow(cv2.cvtColor(contour_view, cv2.COLOR_BGR2RGB))
    axes2[1].set_title("Contours + Boxes")
    axes2[2].imshow(gray, cmap="gray")
    axes2[2].set_title("Reference Gray")
    for axis in axes2:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    save_plot(fig2, folder / f"{stem}_task2_contours.png")
    plt.close(fig2)

    print_block(f"Task 3: Feature Extraction [{stem}]")
    print("Applied ORB feature extractor.")
    print(f"Keypoints found: {len(keypoints)}")
    print(f"Descriptor shape: {None if descriptors is None else descriptors.shape}")

    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle(f"Task 3 - ORB Features [{stem}]", fontsize=15, fontweight="bold")
    axes3[0].imshow(gray, cmap="gray")
    axes3[0].set_title("Gray Image")
    axes3[1].imshow(cv2.cvtColor(orb_view, cv2.COLOR_BGR2RGB))
    axes3[1].set_title("ORB Keypoints")
    for axis in axes3:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    save_plot(fig3, folder / f"{stem}_task3_orb.png")
    plt.close(fig3)

    print_block(f"Task 4: Comparative Analysis [{stem}]")
    print("Edge detector comparison:")
    print("  - Sobel gives thicker gradient-based edges and is easy to interpret.")
    print("  - Canny gives cleaner thin edges and usually works better for object outlines.")
    print("Feature extractor comparison:")
    print("  - ORB is fast and suitable for real-time traffic work.")
    print("  - Keypoints help track cars, lane marks, pedestrians, and scene changes.")
    print("Traffic monitoring use:")
    print("  - Edges help detect road boundaries and lane structure.")
    print("  - Contours help isolate vehicles or pedestrians.")
    print("  - ORB features support matching, tracking, and scene understanding.")


def main():
    args = build_parser().parse_args()
    print("=" * 60)
    print("Welcome to the Traffic Monitoring System")
    print("=" * 60)

    folder = output_dir()
    if args.image:
        process_image(Path(args.image).expanduser(), folder)
    else:
        for image_path in IMAGE_SET.values():
            process_image(image_path, folder)

    print(f"\nDone. Outputs saved in: {folder}")


if __name__ == "__main__":
    main()
