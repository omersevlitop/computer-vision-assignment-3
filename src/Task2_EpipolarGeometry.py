import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
BASE_DIR = Path.cwd().parent
DATASET_DIR = BASE_DIR / "Dataset" / "stereo"
OUTPUT_DIR = BASE_DIR / "outputs" / "task2_epipolar"
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

def detect_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=1500)
    kp, desc = sift.detectAndCompute(gray, None)
    return gray, kp, desc

def match_features(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def draw_epilines(img1, img2, lines, pts1, pts2, max_lines=40):
    """
    Draw epipolar lines on img1 corresponding to pts2 in img2.
    """
    r, c = img1.shape[:2]
    img1_color = img1.copy()
    img2_color = img2.copy()

    if len(lines) > max_lines:
        idx = np.linspace(0, len(lines) - 1, max_lines).astype(int)
        lines = lines[idx]
        pts1 = pts1[idx]
        pts2 = pts2[idx]

    rng = np.random.default_rng(42)

    for line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(int(x) for x in rng.integers(0, 255, size=3))

        a, b, c_line = line
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c_line / b)
            x1, y1 = c, int(-(c_line + a * c) / b)
        else:
            x0 = x1 = int(-c_line / a)
            y0, y1 = 0, r

        cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1_color, tuple(np.int32(pt1)), 4, color, -1)
        cv2.circle(img2_color, tuple(np.int32(pt2)), 4, color, -1)

    return img1_color, img2_color

def resize_to_same_height(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = min(h1, h2)

    img1_resized = cv2.resize(img1, (int(w1 * target_h / h1), target_h))
    img2_resized = cv2.resize(img2, (int(w2 * target_h / h2), target_h))

    return img1_resized, img2_resized

# =========================
# MAIN
# =========================
summary_lines = []
summary_lines.append("Task 2 - Epipolar Geometry and Stereo Rectification")
summary_lines.append("=" * 80)

for pair_name, pair_dir in PAIRS.items():
    print(f"\nProcessing {pair_name}")

    left_img, left_path = read_image_any(pair_dir, "left")
    right_img, right_path = read_image_any(pair_dir, "right")

    left_gray, kp1, desc1 = detect_sift(left_img)
    right_gray, kp2, desc2 = detect_sift(right_img)

    if desc1 is None or desc2 is None:
        print(f"Descriptors could not be computed for {pair_name}")
        continue

    matches = match_features(desc1, desc2, ratio=0.75)

    if len(matches) < 8:
        print(f"Not enough matches for {pair_name}: {len(matches)}")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99
    )

    if F is None or mask is None:
        print(f"Fundamental matrix could not be estimated for {pair_name}")
        continue

    mask = mask.ravel().astype(bool)
    inlier_pts1 = pts1[mask]
    inlier_pts2 = pts2[mask]
    inlier_matches = [m for i, m in enumerate(matches) if mask[i]]

    # Save feature match visualization
    match_vis = cv2.drawMatches(
        left_img, kp1,
        right_img, kp2,
        inlier_matches[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(OUTPUT_DIR / f"{pair_name}_inlier_matches.jpg"), match_vis)

    # Epilines: lines in left image for points in right image
    lines_left = cv2.computeCorrespondEpilines(
        inlier_pts2.reshape(-1, 1, 2),
        2,
        F
    ).reshape(-1, 3)

    epi_left, pts_right_vis = draw_epilines(
        left_img,
        right_img,
        lines_left,
        inlier_pts1,
        inlier_pts2,
        max_lines=40
    )

    epi_pair_left, epi_pair_right = resize_to_same_height(epi_left, pts_right_vis)
    epi_combined = np.hstack((epi_pair_left, epi_pair_right))
    cv2.imwrite(str(OUTPUT_DIR / f"{pair_name}_epipolar_lines.jpg"), epi_combined)

    # Stereo rectification without known intrinsics
    h, w = left_gray.shape[:2]
    ret_rectify, H1, H2 = cv2.stereoRectifyUncalibrated(
        inlier_pts1,
        inlier_pts2,
        F,
        imgSize=(w, h)
    )

    if ret_rectify:
        rect_left = cv2.warpPerspective(left_img, H1, (w, h))
        rect_right = cv2.warpPerspective(right_img, H2, (w, h))

        # Add horizontal guide lines
        rect_left_lines = rect_left.copy()
        rect_right_lines = rect_right.copy()

        for y in range(40, h, 40):
            cv2.line(rect_left_lines, (0, y), (w, y), (0, 255, 0), 1)
            cv2.line(rect_right_lines, (0, y), (w, y), (0, 255, 0), 1)

        rect_left_resized, rect_right_resized = resize_to_same_height(rect_left_lines, rect_right_lines)
        rectified_combined = np.hstack((rect_left_resized, rect_right_resized))
        cv2.imwrite(str(OUTPUT_DIR / f"{pair_name}_rectified.jpg"), rectified_combined)
    else:
        print(f"Rectification failed for {pair_name}")

    summary_lines.append(f"\n{pair_name}")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Left image: {left_path.name}")
    summary_lines.append(f"Right image: {right_path.name}")
    summary_lines.append(f"Detected keypoints left: {len(kp1)}")
    summary_lines.append(f"Detected keypoints right: {len(kp2)}")
    summary_lines.append(f"Ratio-test matches: {len(matches)}")
    summary_lines.append(f"RANSAC inliers: {len(inlier_pts1)}")
    summary_lines.append(f"Inlier ratio: {len(inlier_pts1) / len(matches):.4f}")
    summary_lines.append("Fundamental matrix:")
    summary_lines.append(np.array2string(F, precision=6, suppress_small=True))
    summary_lines.append(f"Rectification success: {ret_rectify}")

# Save summary
summary_path = OUTPUT_DIR / "task2_epipolar_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    for line in summary_lines:
        f.write(line + "\n")

print("\nTask 2 completed.")
print("Outputs saved to:", OUTPUT_DIR)
print("Summary file:", summary_path)