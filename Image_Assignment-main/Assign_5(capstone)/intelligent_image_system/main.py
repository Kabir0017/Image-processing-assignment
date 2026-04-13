# Name: Kabir Bidhuri
# Roll No: 2301010461
# Course Name: BTech CSE CORE
# Assignment Title: Intelligent Image Processing Capstone
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


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser():
    parser = ArgumentParser(description="End-to-end intelligent image processing pipeline.")
    parser.add_argument("--image", type=str, default=None, help="Path to one image")
    parser.add_argument("--folder", type=str, default=None, help="Path to image folder")
    parser.add_argument("--webcam", action="store_true", help="Capture one frame from webcam")
    return parser


def create_output_dir():
    out_dir = SCRIPT_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def print_header():
    print("=" * 70)
    print("Welcome to the Intelligent Image Processing System")
    print("This pipeline performs preprocessing, restoration, segmentation,")
    print("feature extraction, and objective quality evaluation.")
    print("=" * 70)


def capture_from_webcam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Webcam could not be opened.")
    ok, frame = cam.read()
    cam.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read frame from webcam.")
    return frame


def pick_file_dialog(title, filetypes):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
    except Exception as exc:
        raise RuntimeError(
            "File picker could not be opened. Run with --image PATH instead."
        ) from exc

    if not selected:
        raise FileNotFoundError("No file was selected.")
    return Path(selected)


def pick_folder_dialog(title):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title=title)
        root.destroy()
    except Exception as exc:
        raise RuntimeError(
            "Folder picker could not be opened. Run with --folder PATH instead."
        ) from exc

    if not selected:
        raise FileNotFoundError("No folder was selected.")
    return Path(selected)


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def images_in_folder(folder_path):
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_paths = []
    for item in sorted(folder_path.iterdir()):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(item)

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in folder: {folder_path}")
    return image_paths


def ask_user_input_mode():
    print("\nChoose input mode:")
    print("1. Single image upload")
    print("2. Folder upload (all images in folder)")
    print("3. Webcam capture")
    choice = input("Enter 1/2/3: ").strip()

    if choice == "1":
        image_path = pick_file_dialog(
            "Select an image",
            [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        return "image", image_path
    if choice == "2":
        folder_path = pick_folder_dialog("Select a folder containing images")
        return "folder", folder_path
    if choice == "3":
        return "webcam", None

    raise ValueError("Invalid choice. Please run again and enter 1, 2, or 3.")


def handle_empty_selection(message):
    print(f"\n{message}")
    print("No file or folder was uploaded. Ending run.")
    return None, None


def preprocess(image):
    resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized, gray


def add_gaussian_noise(gray, sigma=18.0, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, gray.shape).astype(np.float32)
    noisy = gray.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(gray, amount=0.035, seed=42):
    rng = np.random.default_rng(seed)
    noisy = gray.copy()
    total = gray.size
    salt = int(np.ceil(total * amount * 0.5))
    pepper = int(np.ceil(total * amount * 0.5))

    salt_coords = tuple(rng.integers(0, dim, salt) for dim in gray.shape)
    pepper_coords = tuple(rng.integers(0, dim, pepper) for dim in gray.shape)
    noisy[salt_coords] = 255
    noisy[pepper_coords] = 0
    return noisy


def apply_restoration_filters(noisy):
    return {
        "mean_filter": cv2.blur(noisy, (5, 5)),
        "median_filter": cv2.medianBlur(noisy, 5),
        "gaussian_filter": cv2.GaussianBlur(noisy, (5, 5), 1.2),
    }


def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def segment_image(gray):
    global_threshold = int(np.mean(gray))
    _, global_binary = cv2.threshold(gray, global_threshold, 255, cv2.THRESH_BINARY)
    otsu_threshold, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return global_threshold, global_binary, int(otsu_threshold), otsu_binary


def morphology(binary):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return dilated, eroded


def sobel_and_canny(gray):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))
    canny = cv2.Canny(gray, 80, 160)
    return sobel_mag, canny


def contour_representation(color, gray):
    image_h, image_w = gray.shape[:2]
    image_area = float(image_h * image_w)

    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_map = cv2.Canny(smooth, 60, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    detection_mask = cv2.dilate(edge_map, kernel, iterations=1)

    contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing = color.copy()
    measured = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 60.0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        box_area = float(w * h)
        fill_ratio = area / max(box_area, 1.0)

        if box_area > image_area * 0.30:
            continue
        if w < 8 or h < 8:
            continue
        if fill_ratio < 0.03:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        measured.append((area, perimeter, x, y, w, h, contour))

    measured.sort(key=lambda item: item[0], reverse=True)

    trimmed = []
    for rank, (area, perimeter, x, y, w, h, contour) in enumerate(measured[:20], start=1):
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(drawing, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            drawing,
            f"Obj {rank}",
            (x, max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        trimmed.append((area, perimeter, x, y, w, h))

    return drawing, trimmed, detection_mask


def orb_features(color, gray):
    orb = cv2.ORB_create(nfeatures=600)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    marked = cv2.drawKeypoints(
        color,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return keypoints, descriptors, marked


def mse(reference, target):
    diff = reference.astype(np.float32) - target.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr(reference, target):
    error = mse(reference, target)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / error))


def ssim(reference, target):
    ref = reference.astype(np.float64)
    tar = target.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(ref, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(tar, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = cv2.filter2D(ref * ref, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(tar * tar, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(ref * tar, -1, window)[5:-5, 5:-5] - mu12

    score_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(score_map.mean())


def metric_report(reference, target, title):
    m = mse(reference, target)
    p = psnr(reference, target)
    s = ssim(reference, target)
    print(f"{title}: MSE={m:.2f}, PSNR={p:.2f} dB, SSIM={s:.4f}")
    return m, p, s


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def make_stage_figures(
    tag,
    out_dir,
    color,
    gray,
    gaussian_noisy,
    sp_noisy,
    restored_best,
    enhanced,
    global_binary,
    otsu_binary,
    global_dilated,
    global_eroded,
    otsu_dilated,
    otsu_eroded,
    sobel,
    canny,
    detection_mask,
    contour_img,
    orb_img,
    descriptors,
):
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5))
    fig1.suptitle(f"Task 2 - Acquisition & Preprocessing [{tag}]", fontsize=14, fontweight="bold")
    axes1[0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    axes1[0].set_title("Original")
    axes1[1].imshow(gray, cmap="gray")
    axes1[1].set_title("Grayscale")
    for axis in axes1:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    save_figure(fig1, out_dir / f"{tag}_task2_preprocessing.png")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9))
    fig2.suptitle(f"Task 3 - Enhancement & Restoration [{tag}]", fontsize=14, fontweight="bold")
    axes2[0, 0].imshow(gaussian_noisy, cmap="gray")
    axes2[0, 0].set_title("Gaussian Noise")
    axes2[0, 1].imshow(sp_noisy, cmap="gray")
    axes2[0, 1].set_title("Salt-Pepper Noise")
    axes2[0, 2].imshow(restored_best, cmap="gray")
    axes2[0, 2].set_title("Best Restored")
    axes2[1, 0].imshow(enhanced, cmap="gray")
    axes2[1, 0].set_title("CLAHE Enhanced")
    axes2[1, 1].imshow(gray, cmap="gray")
    axes2[1, 1].set_title("Reference Gray")
    axes2[1, 2].axis("off")
    for axis in axes2.flat:
        if axis.has_data():
            axis.set_xticks([])
            axis.set_yticks([])
    plt.tight_layout()
    save_figure(fig2, out_dir / f"{tag}_task3_restoration_enhancement.png")
    plt.close(fig2)

    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 9))
    fig3.suptitle(f"Task 4 - Segmentation & Morphology [{tag}]", fontsize=14, fontweight="bold")
    axes3[0, 0].imshow(global_binary, cmap="gray")
    axes3[0, 0].set_title("Global Threshold")
    axes3[0, 1].imshow(global_dilated, cmap="gray")
    axes3[0, 1].set_title("Global Dilation")
    axes3[0, 2].imshow(global_eroded, cmap="gray")
    axes3[0, 2].set_title("Global Erosion")
    axes3[1, 0].imshow(otsu_binary, cmap="gray")
    axes3[1, 0].set_title("Otsu Threshold")
    axes3[1, 1].imshow(otsu_dilated, cmap="gray")
    axes3[1, 1].set_title("Otsu Dilation")
    axes3[1, 2].imshow(otsu_eroded, cmap="gray")
    axes3[1, 2].set_title("Otsu Erosion")
    for axis in axes3.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    save_figure(fig3, out_dir / f"{tag}_task4_segmentation_morphology.png")
    plt.close(fig3)

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle(f"Task 5 - Edge Detection [{tag}]", fontsize=14, fontweight="bold")
    axes4[0].imshow(sobel, cmap="gray")
    axes4[0].set_title("Sobel")
    axes4[1].imshow(canny, cmap="gray")
    axes4[1].set_title("Canny")
    for axis in axes4:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    save_figure(fig4, out_dir / f"{tag}_task5_edges.png")
    plt.close(fig4)

    fig5, axes5 = plt.subplots(1, 4, figsize=(20, 5))
    fig5.suptitle(f"Task 5 - Objects & ORB Features [{tag}]", fontsize=14, fontweight="bold")
    axes5[0].imshow(detection_mask, cmap="gray")
    axes5[0].set_title("Detection Mask")
    axes5[1].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes5[1].set_title("Contours + Bounding Boxes")
    axes5[2].imshow(cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB))
    axes5[2].set_title("ORB Keypoints")
    if descriptors is not None and descriptors.size > 0:
        preview = descriptors[:40, :]
        axes5[3].imshow(preview, cmap="viridis", aspect="auto")
        axes5[3].set_title("ORB Descriptors")
    else:
        axes5[3].axis("off")
        axes5[3].text(0.1, 0.5, "No descriptors", fontsize=12)
    for axis in axes5:
        if axis.has_data():
            axis.set_xticks([])
            axis.set_yticks([])
    plt.tight_layout()
    save_figure(fig5, out_dir / f"{tag}_task5_objects_features.png")
    plt.close(fig5)

    fig6, axes6 = plt.subplots(2, 3, figsize=(15, 9))
    fig6.suptitle(f"Task 7 - Final Pipeline [{tag}]", fontsize=14, fontweight="bold")
    axes6[0, 0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    axes6[0, 0].set_title("Original")
    axes6[0, 1].imshow(sp_noisy, cmap="gray")
    axes6[0, 1].set_title("Noisy")
    axes6[0, 2].imshow(restored_best, cmap="gray")
    axes6[0, 2].set_title("Restored")
    axes6[1, 0].imshow(enhanced, cmap="gray")
    axes6[1, 0].set_title("Enhanced")
    axes6[1, 1].imshow(otsu_dilated, cmap="gray")
    axes6[1, 1].set_title("Segmented")
    axes6[1, 2].imshow(cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB))
    axes6[1, 2].set_title("Features")
    for axis in axes6.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    save_figure(fig6, out_dir / f"{tag}_task7_pipeline_overview.png")
    plt.close(fig6)


def process_pipeline(image, tag, out_dir):
    print("\n" + "=" * 70)
    print(f"Processing Image: {tag}")
    print("=" * 70)

    print("Task 2: Image acquisition and preprocessing...")
    color, gray = preprocess(image)

    print("Task 3: Adding noise, restoring, and enhancing contrast...")
    gaussian_noisy = add_gaussian_noise(gray, sigma=18.0, seed=42)
    sp_noisy = add_salt_pepper_noise(gray, amount=0.035, seed=42)

    restored_from_gaussian = apply_restoration_filters(gaussian_noisy)
    restored_from_sp = apply_restoration_filters(sp_noisy)

    restored_pool = {
        "gaussian_mean": restored_from_gaussian["mean_filter"],
        "gaussian_median": restored_from_gaussian["median_filter"],
        "gaussian_gaussian": restored_from_gaussian["gaussian_filter"],
        "sp_mean": restored_from_sp["mean_filter"],
        "sp_median": restored_from_sp["median_filter"],
        "sp_gaussian": restored_from_sp["gaussian_filter"],
    }
    best_restored_name, best_restored = max(
        restored_pool.items(), key=lambda item: psnr(gray, item[1])
    )
    enhanced = enhance_contrast(best_restored)

    print("Task 4: Segmentation and morphological refinement...")
    global_t, global_binary, otsu_t, otsu_binary = segment_image(enhanced)
    global_dilated, global_eroded = morphology(global_binary)
    otsu_dilated, otsu_eroded = morphology(otsu_binary)

    print("Task 5: Edge detection, object representation, and ORB features...")
    sobel, canny = sobel_and_canny(enhanced)
    contour_img, measurements, detection_mask = contour_representation(color, gray)
    keypoints, descriptors, orb_img = orb_features(color, enhanced)

    print("Task 6: Computing MSE, PSNR, SSIM...")
    metric_report(gray, enhanced, "Original vs Enhanced")
    metric_report(gray, best_restored, "Original vs Restored")

    print(f"Detected objects: {len(measurements)}")
    print("Top detected objects (area, perimeter):")
    if measurements:
        for idx, (area, perimeter, x, y, w, h) in enumerate(measurements[:8], start=1):
            print(
                f"  {idx}. Area={area:.1f}, Perimeter={perimeter:.1f}, "
                f"Box=({x}, {y}, {w}, {h})"
            )
    else:
        print("  No major contours found with current thresholding.")

    print("Edge and feature summary:")
    print(f"  Best restored output: {best_restored_name}")
    print(f"  Global threshold: {global_t}, Otsu threshold: {otsu_t}")
    print(f"  Sobel edge pixels (>40): {int(np.count_nonzero(sobel > 40))}")
    print(f"  Canny edge pixels: {int(np.count_nonzero(canny))}")
    print(f"  ORB keypoints: {len(keypoints)}")
    print(f"  Descriptor shape: {None if descriptors is None else descriptors.shape}")

    print("Task 7: Saving all visual stages...")
    make_stage_figures(
        tag,
        out_dir,
        color,
        gray,
        gaussian_noisy,
        sp_noisy,
        best_restored,
        enhanced,
        global_binary,
        otsu_binary,
        global_dilated,
        global_eroded,
        otsu_dilated,
        otsu_eroded,
        sobel,
        canny,
        detection_mask,
        contour_img,
        orb_img,
        descriptors,
    )

    print("Conclusion:")
    print("  Pipeline successfully converted a raw frame into cleaned, segmented,")
    print("  and feature-rich output suitable for monitoring and analysis tasks.")


def main():
    args = build_parser().parse_args()
    print_header()
    out_dir = create_output_dir()

    selected_flags = int(args.image is not None) + int(args.folder is not None) + int(args.webcam)
    if selected_flags > 1:
        raise ValueError("Use only one input option at a time: --image OR --folder OR --webcam")

    if args.webcam:
        print("Using webcam as image source...")
        frame = capture_from_webcam()
        process_pipeline(frame, "webcam_capture", out_dir)
    elif args.image:
        img_path = Path(args.image).expanduser()
        process_pipeline(load_image(img_path), img_path.stem.replace(" ", "_"), out_dir)
    elif args.folder:
        folder_path = Path(args.folder).expanduser()
        print(f"Processing all images from folder: {folder_path}")
        for path in images_in_folder(folder_path):
            process_pipeline(load_image(path), path.stem.replace(" ", "_"), out_dir)
    else:
        try:
            mode, target = ask_user_input_mode()
        except (FileNotFoundError, RuntimeError) as exc:
            handle_empty_selection(str(exc))
            return

        if mode == "webcam":
            frame = capture_from_webcam()
            process_pipeline(frame, "webcam_capture", out_dir)
        elif mode == "image":
            process_pipeline(load_image(target), target.stem.replace(" ", "_"), out_dir)
        else:
            print(f"Processing all images from folder: {target}")
            for path in images_in_folder(target):
                process_pipeline(load_image(path), path.stem.replace(" ", "_"), out_dir)

    print("\nAll processing complete.")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()







