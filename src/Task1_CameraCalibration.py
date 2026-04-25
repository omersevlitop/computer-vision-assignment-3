import cv2
import numpy as np
from pathlib import Path
import os

# =========================
# PATHS
# =========================
BASE_DIR = Path(r"D:\Computer Vision Assignment 3")
DATASET_DIR = BASE_DIR / "Dataset" / "calibration"
OUTPUT_DIR = BASE_DIR / "outputs" / "task1_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Dataset path:", DATASET_DIR)
print("Folder exists:", DATASET_DIR.exists())

# =========================
# IMAGE FILE SEARCH
# =========================
image_paths = sorted(DATASET_DIR.glob("*.jpg"))

image_paths = sorted(image_paths)

print("\nFound files:")
for p in image_paths:
    print("-", p.name)

if len(image_paths) < 8:
    raise RuntimeError(f"At least 8 calibration images are required. Found: {len(image_paths)}")

# =========================
# CHECKERBOARD SETTINGS
# =========================
CHECKERBOARD = (9, 6)  # OpenCV left01-left14 images usually use 9x6 inner corners

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []
successful_images = []
gray_shape = None

# =========================
# CHESSBOARD CORNER DETECTION
# =========================
for img_path in image_paths:
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"Could not read: {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    print(f"{img_path.name}: corners found = {ret}")

    if ret:
        objpoints.append(objp)

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        imgpoints.append(corners_refined)
        successful_images.append(img_path.name)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_refined, ret)
        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_corners.jpg"), vis)

if len(objpoints) < 8:
    raise RuntimeError(
        f"Calibration failed: at least 8 successful checkerboard detections required. "
        f"Successful detections: {len(objpoints)}"
    )

# =========================
# CAMERA CALIBRATION
# =========================
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray_shape,
    None,
    None
)

# =========================
# REPROJECTION ERROR
# =========================
total_error = 0.0
per_image_errors = []

for i in range(len(objpoints)):
    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
    per_image_errors.append(error)
    total_error += error

mean_error = total_error / len(objpoints)

# =========================
# UNDISTORTION EXAMPLE
# =========================
example_path = DATASET_DIR / successful_images[0]
example_img = cv2.imread(str(example_path))

h, w = example_img.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    1,
    (w, h)
)

undistorted = cv2.undistort(
    example_img,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)

cv2.imwrite(str(OUTPUT_DIR / "original_example.jpg"), example_img)
cv2.imwrite(str(OUTPUT_DIR / "undistorted_example.jpg"), undistorted)

comparison = np.hstack((example_img, undistorted))
cv2.imwrite(str(OUTPUT_DIR / "undistortion_comparison.jpg"), comparison)

# =========================
# SAVE RESULTS
# =========================
results_path = OUTPUT_DIR / "task1_calibration_results.txt"

with open(results_path, "w", encoding="utf-8") as f:
    f.write("Task 1 - Camera Calibration Results\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset path: {DATASET_DIR}\n")
    f.write(f"Number of input images: {len(image_paths)}\n")
    f.write(f"Successful checkerboard detections: {len(objpoints)}\n")
    f.write(f"Checkerboard inner corners: {CHECKERBOARD}\n\n")

    f.write("Successful images:\n")
    for name in successful_images:
        f.write(f"- {name}\n")

    f.write("\nCamera Matrix:\n")
    f.write(np.array2string(camera_matrix, precision=6, suppress_small=True))

    f.write("\n\nDistortion Coefficients:\n")
    f.write(np.array2string(dist_coeffs, precision=6, suppress_small=True))

    f.write("\n\nMean Reprojection Error:\n")
    f.write(f"{mean_error:.6f}\n\n")

    f.write("Per-image Reprojection Errors:\n")
    for name, err in zip(successful_images, per_image_errors):
        f.write(f"{name}: {err:.6f}\n")

# =========================
# PRINT RESULTS
# =========================
print("\nTask 1 completed successfully.")
print("Successful detections:", len(objpoints))
print("\nCamera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)
print("\nMean reprojection error:", mean_error)
print("\nOutputs saved to:", OUTPUT_DIR)
print("Results file:", results_path)