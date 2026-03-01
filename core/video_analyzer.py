import cv2

from utils.logger import logger


def analyze_video_dynamics(video_path: str, start_time: float, end_time: float) -> float:
    """
    Calculates the average frame difference for a specific time window.
    High frame difference indicates fast motion / heavy action.
    This is computationally expensive, so it should only be run on short candidate segments.
    """
    logger.info(f"Analyzing dynamics for {video_path} from {start_time:.2f}s to {end_time:.2f}s...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video {video_path}")
        return 0.0

    # Seek to start time (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

    prev_frame = None
    total_diff = 0.0
    frame_count = 0

    while cap.isOpened():
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_msec > end_time * 1000.0:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale to speed up difference calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate absolute difference between current and previous frame
            diff = cv2.absdiff(gray, prev_frame)
            # Mean pixel difference
            mean_diff = diff.mean()
            total_diff += mean_diff
            frame_count += 1

        prev_frame = gray

    cap.release()

    if frame_count == 0:
        return 0.0

    avg_diff = total_diff / frame_count
    logger.debug(f"Dynamics score for [{start_time:.2f}-{end_time:.2f}]: {avg_diff:.2f}")
    return avg_diff
