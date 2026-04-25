import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
BASE_DIR = Path.cwd().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "task5_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATHS = {
    "aloe_left": BASE_DIR / "Dataset" / "stereo" / "pair1" / "left.jpg",
    "cones_left": BASE_DIR / "Dataset" / "stereo" / "pair2" / "left.png",
    "flow_frame": BASE_DIR / "outputs" / "task4_optical_flow" / "frame_00.jpg",
}

# =========================
# HELPERS
# =========================
def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def otsu_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold_value, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    result = cv2.bitwise_and(img, img, mask=mask)
    return mask, result, threshold_value

def kmeans_segmentation(img, k=3):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(rgb.shape)
    segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)

    label_map = labels.reshape((img.shape[0], img.shape[1]))
    label_vis = np.uint8(255 * label_map / max(1, k - 1))

    return segmented_bgr, label_vis

def create_comparison(name, img, otsu_mask, otsu_result, kmeans3, kmeans5, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Task 5 - Classical Segmentation: {name}", fontsize=16)

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(otsu_mask, cmap="gray")
    axes[1].set_title("Otsu Mask")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(otsu_result, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Otsu Result")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(kmeans3, cv2.COLOR_BGR2RGB))
    axes[3].set_title("K-means k=3")
    axes[3].axis("off")

    axes[4].imshow(cv2.cvtColor(kmeans5, cv2.COLOR_BGR2RGB))
    axes[4].set_title("K-means k=5")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# MAIN
# =========================
summary_lines = []
summary_lines.append("Task 5 - Classical Segmentation Results")
summary_lines.append("=" * 80)

for name, path in IMAGE_PATHS.items():
    print(f"Processing: {name}")

    img = read_image(path)

    # resize large images for consistent display and faster processing
    max_width = 800
    h, w = img.shape[:2]
    if w > max_width:
        new_h = int(h * max_width / w)
        img = cv2.resize(img, (max_width, new_h))

    otsu_mask, otsu_result, threshold_value = otsu_segmentation(img)
    kmeans3, label3 = kmeans_segmentation(img, k=3)
    kmeans5, label5 = kmeans_segmentation(img, k=5)

    cv2.imwrite(str(OUTPUT_DIR / f"{name}_otsu_mask.jpg"), otsu_mask)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_otsu_result.jpg"), otsu_result)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_kmeans_k3.jpg"), kmeans3)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_kmeans_k5.jpg"), kmeans5)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_kmeans_k3_labels.jpg"), label3)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_kmeans_k5_labels.jpg"), label5)

    comparison_path = OUTPUT_DIR / f"{name}_comparison.png"
    create_comparison(
        name,
        img,
        otsu_mask,
        otsu_result,
        kmeans3,
        kmeans5,
        comparison_path
    )

    # Simple quantitative properties without ground truth
    otsu_foreground_ratio = np.count_nonzero(otsu_mask) / otsu_mask.size

    summary_lines.append(f"\nImage: {name}")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Input path: {path}")
    summary_lines.append(f"Otsu threshold value: {threshold_value:.2f}")
    summary_lines.append(f"Otsu foreground pixel ratio: {otsu_foreground_ratio:.4f}")
    summary_lines.append("K-means settings: k=3 and k=5")
    summary_lines.append("Ground truth mask: not available; qualitative evaluation used.")

summary_path = OUTPUT_DIR / "task5_segmentation_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    for line in summary_lines:
        f.write(line + "\n")

print("Task 5 completed.")
print("Outputs saved to:", OUTPUT_DIR)
print("Summary:", summary_path)