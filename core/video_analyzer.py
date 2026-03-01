import cv2
import numpy as np

from utils.logger import logger


def analyze_video_dynamics(video_path: str, start_time: float, end_time: float) -> dict:
    """
    Calculates advanced dynamics for Anime:
    - avg_diff: General motion intensity.
    - max_diff: Peak action intensity (Spacing).
    - effective_fps: Ratio of frames with significant motion (Sakuga/Ones vs Threes).
    - impact_score: Detects sudden visual spikes (Impact Frames).
    """
    logger.info(
        f"Analyzing anime dynamics for {video_path} "
        f"from {start_time:.2f}s to {end_time:.2f}s..."
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video {video_path}")
        return {"avg_diff": 0.0, "max_diff": 0.0, "effective_fps": 0.0, "impact_score": 0.0}

    # Seek to start time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

    prev_frame = None
    diffs = []
    brightness_list = []

    # Threshold for "significant change" to count towards effective FPS
    # This is a heuristic value for pixel-wise mean difference (0-255)
    MOTION_THRESHOLD = 2.0

    while cap.isOpened():
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_msec > end_time * 1000.0:
            break

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        brightness_list.append(mean_brightness)

        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = float(diff.mean())
            diffs.append(mean_diff)

        prev_frame = gray

    cap.release()

    if not diffs:
        return {"avg_diff": 0.0, "max_diff": 0.0, "effective_fps": 0.0, "impact_score": 0.0}

    avg_diff = np.mean(diffs)
    max_diff = np.max(diffs)

    # Effective FPS: How many frames are actually "moving" (Ones vs Threes logic)
    significant_frames = sum(1 for d in diffs if d > MOTION_THRESHOLD)
    effective_fps_ratio = significant_frames / len(diffs)

    # Impact Frame Detection: Sudden spikes in frame difference
    # An impact frame usually has a much higher difference than its neighbors
    impact_count = 0
    if len(diffs) > 3:
        for i in range(1, len(diffs) - 1):
            # If current diff is 2.5x the average of neighbors, it's a spike
            neighbor_avg = (diffs[i - 1] + diffs[i + 1]) / 2.0
            if diffs[i] > neighbor_avg * 2.5 and diffs[i] > 5.0:
                impact_count += 1

    # Also check brightness spikes (flashes)
    brightness_spikes = 0
    if len(brightness_list) > 3:
        for i in range(1, len(brightness_list) - 1):
            b_diff = abs(brightness_list[i] - brightness_list[i - 1])
            if b_diff > 30:  # Significant brightness jump (0-255 scale)
                brightness_spikes += 1

    results = {
        "avg_diff": float(avg_diff),
        "max_diff": float(max_diff),
        "effective_fps": float(effective_fps_ratio * 100.0),  # Normalize to 0-100
        "impact_score": float((impact_count + brightness_spikes) * 10.0),  # Heuristic scaling
    }

    logger.debug(
        f"Dynamics: Avg={avg_diff:.2f}, Max={max_diff:.2f}, "
        f"EffFPS={results['effective_fps']:.1f}%, Impact={results['impact_score']:.1f}"
    )
    return results
