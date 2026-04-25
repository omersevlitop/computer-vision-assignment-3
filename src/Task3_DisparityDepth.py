import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
BASE_DIR = Path.cwd().parent
DATASET_DIR = BASE_DIR / "Dataset" / "stereo"
OUTPUT_DIR = BASE_DIR / "outputs" / "task3_disparity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "pair1_aloe": DATASET_DIR / "pair1",
    "pair2_cones": DATASET_DIR / "pair2",
}

# =========================
# HELPERS
# =========================
def read_image_any(pair_dir, name):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        path = pair_dir / f"{name}{ext}"
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                return img, path
    raise FileNotFoundError(f"{name} image not found in {pair_dir}")

def normalize_disparity(disp):
    disp = disp.astype(np.float32)
    valid = disp > disp.min()
    if np.any(valid):
        disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    else:
        disp_norm = np.zeros_like(disp)
    return np.uint8(disp_norm)

def save_colormap(path, disp):
    disp_norm = normalize_disparity(disp)
    color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), color)
    return color

def create_comparison(left, right, bm_disp_color, sgbm_disp_color, title, save_path):
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    bm_rgb = cv2.cvtColor(bm_disp_color, cv2.COLOR_BGR2RGB)
    sgbm_rgb = cv2.cvtColor(sgbm_disp_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)

    axes[0].imshow(left_rgb)
    axes[0].set_title("Left Image")
    axes[0].axis("off")

    axes[1].imshow(right_rgb)
    axes[1].set_title("Right Image")
    axes[1].axis("off")

    axes[2].imshow(bm_rgb)
    axes[2].set_title("StereoBM Disparity")
    axes[2].axis("off")

    axes[3].imshow(sgbm_rgb)
    axes[3].set_title("StereoSGBM Disparity")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close()

def disparity_stats(disp):
    disp = disp.astype(np.float32)
    valid = disp[disp > 0]

    if valid.size == 0:
        return {
            "valid_pixels": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "valid_pixels": int(valid.size),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }

# =========================
# MAIN
# =========================
summary_lines = []
summary_lines.append("Task 3 - Disparity and Depth Estimation")
summary_lines.append("=" * 80)

csv_lines = []
csv_lines.append("pair,method,num_disparities,block_size,valid_pixels,mean_disparity,std_disparity,min_disparity,max_disparity")

for pair_name, pair_dir in PAIRS.items():
    print(f"\nProcessing {pair_name}")

    left_img, left_path = read_image_any(pair_dir, "left")
    right_img, right_path = read_image_any(pair_dir, "right")

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Ensure same size
    h = min(left_gray.shape[0], right_gray.shape[0])
    w = min(left_gray.shape[1], right_gray.shape[1])

    left_img = cv2.resize(left_img, (w, h))
    right_img = cv2.resize(right_img, (w, h))
    left_gray = cv2.resize(left_gray, (w, h))
    right_gray = cv2.resize(right_gray, (w, h))

    # Two parameter settings
    settings = [
        {"num_disp": 64, "block_size": 9},
        {"num_disp": 128, "block_size": 15},
    ]

    for idx, params in enumerate(settings, start=1):
        num_disp = params["num_disp"]
        block_size = params["block_size"]

        # =========================
        # StereoBM
        # =========================
        stereo_bm = cv2.StereoBM_create(
            numDisparities=num_disp,
            blockSize=block_size
        )

        bm_disp = stereo_bm.compute(left_gray, right_gray).astype(np.float32) / 16.0

        bm_color = save_colormap(
            OUTPUT_DIR / f"{pair_name}_setting{idx}_StereoBM_disparity_color.jpg",
            bm_disp
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{pair_name}_setting{idx}_StereoBM_disparity_gray.jpg"),
            normalize_disparity(bm_disp)
        )

        bm_stats = disparity_stats(bm_disp)

        csv_lines.append(
            f"{pair_name},StereoBM,{num_disp},{block_size},"
            f"{bm_stats['valid_pixels']},{bm_stats['mean']:.4f},{bm_stats['std']:.4f},"
            f"{bm_stats['min']:.4f},{bm_stats['max']:.4f}"
        )

        # =========================
        # StereoSGBM
        # =========================
        stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        sgbm_disp = stereo_sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0

        sgbm_color = save_colormap(
            OUTPUT_DIR / f"{pair_name}_setting{idx}_StereoSGBM_disparity_color.jpg",
            sgbm_disp
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{pair_name}_setting{idx}_StereoSGBM_disparity_gray.jpg"),
            normalize_disparity(sgbm_disp)
        )

        sgbm_stats = disparity_stats(sgbm_disp)

        csv_lines.append(
            f"{pair_name},StereoSGBM,{num_disp},{block_size},"
            f"{sgbm_stats['valid_pixels']},{sgbm_stats['mean']:.4f},{sgbm_stats['std']:.4f},"
            f"{sgbm_stats['min']:.4f},{sgbm_stats['max']:.4f}"
        )

        # =========================
        # Save comparison figure
        # =========================
        comparison_path = OUTPUT_DIR / f"{pair_name}_setting{idx}_comparison.png"

        create_comparison(
            left_img,
            right_img,
            bm_color,
            sgbm_color,
            title=f"{pair_name} - Disparity Estimation Setting {idx} "
                  f"(numDisp={num_disp}, blockSize={block_size})",
            save_path=comparison_path
        )

        summary_lines.append(f"\n{pair_name} - Setting {idx}")
        summary_lines.append("-" * 80)
        summary_lines.append(f"numDisparities: {num_disp}")
        summary_lines.append(f"blockSize: {block_size}")

        summary_lines.append("StereoBM statistics:")
        summary_lines.append(f"  valid pixels: {bm_stats['valid_pixels']}")
        summary_lines.append(f"  mean disparity: {bm_stats['mean']:.4f}")
        summary_lines.append(f"  std disparity: {bm_stats['std']:.4f}")
        summary_lines.append(f"  min disparity: {bm_stats['min']:.4f}")
        summary_lines.append(f"  max disparity: {bm_stats['max']:.4f}")

        summary_lines.append("StereoSGBM statistics:")
        summary_lines.append(f"  valid pixels: {sgbm_stats['valid_pixels']}")
        summary_lines.append(f"  mean disparity: {sgbm_stats['mean']:.4f}")
        summary_lines.append(f"  std disparity: {sgbm_stats['std']:.4f}")
        summary_lines.append(f"  min disparity: {sgbm_stats['min']:.4f}")
        summary_lines.append(f"  max disparity: {sgbm_stats['max']:.4f}")

# Save summary
summary_path = OUTPUT_DIR / "task3_disparity_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    for line in summary_lines:
        f.write(line + "\n")

# Save CSV
csv_path = OUTPUT_DIR / "task3_disparity_metrics.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    for line in csv_lines:
        f.write(line + "\n")

print("\nTask 3 completed.")
print("Outputs saved to:", OUTPUT_DIR)
print("Summary file:", summary_path)
print("Metrics CSV:", csv_path)