import cv2
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================
BASE_DIR = Path.cwd().parent
VIDEO_PATH = BASE_DIR / "Dataset" / "optical_flow" / "flow_video.mp4"
OUTPUT_DIR = BASE_DIR / "outputs" / "task4_optical_flow"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# SETTINGS
# =========================
MAX_FRAMES = 8
FRAME_STEP = 5
ARROW_STEP = 20

# =========================
# HELPERS
# =========================
def draw_flow_arrows(frame, flow, step=20):
    h, w = frame.shape[:2]
    vis = frame.copy()

    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

    return vis

def flow_to_color(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# =========================
# READ VIDEO
# =========================
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Video could not be opened: {VIDEO_PATH}")

frames = []
frame_index = 0

while len(frames) < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % FRAME_STEP == 0:
        frame = cv2.resize(frame, (640, 360))
        frames.append(frame)

    frame_index += 1

cap.release()

if len(frames) < 5:
    raise RuntimeError(f"At least 5 frames are required. Extracted only {len(frames)} frames.")

# Save extracted frames
for i, frame in enumerate(frames):
    cv2.imwrite(str(OUTPUT_DIR / f"frame_{i:02d}.jpg"), frame)

summary_lines = []
summary_lines.append("Task 4 - Optical Flow Results")
summary_lines.append("=" * 80)
summary_lines.append(f"Video path: {VIDEO_PATH}")
summary_lines.append(f"Extracted frames: {len(frames)}")
summary_lines.append(f"Frame step: {FRAME_STEP}")
summary_lines.append("")
summary_lines.append("Farneback parameters:")
summary_lines.append("pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2")
summary_lines.append("")

# =========================
# FARNEBACK OPTICAL FLOW
# =========================
for i in range(len(frames) - 1):
    prev = frames[i]
    nxt = frames[i + 1]

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    arrow_vis = draw_flow_arrows(prev, flow, step=ARROW_STEP)
    color_vis = flow_to_color(flow)

    cv2.imwrite(str(OUTPUT_DIR / f"flow_{i:02d}_arrows.jpg"), arrow_vis)
    cv2.imwrite(str(OUTPUT_DIR / f"flow_{i:02d}_color.jpg"), color_vis)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(np.mean(magnitude))
    max_mag = float(np.max(magnitude))

    summary_lines.append(f"Frame pair {i} -> {i+1}")
    summary_lines.append(f"  Mean flow magnitude: {mean_mag:.4f}")
    summary_lines.append(f"  Max flow magnitude: {max_mag:.4f}")
    summary_lines.append("-" * 80)

# Save summary
summary_path = OUTPUT_DIR / "task4_optical_flow_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    for line in summary_lines:
        f.write(line + "\n")

print("Task 4 completed.")
print("Extracted frames:", len(frames))
print("Outputs saved to:", OUTPUT_DIR)
print("Summary:", summary_path)