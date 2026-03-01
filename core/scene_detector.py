import re
import subprocess

from utils.logger import logger


def detect_scene_changes(video_path: str, threshold: float = 0.3) -> list[float]:
    """
    Detects scene changes (hard cuts) in the video using FFmpeg.
    Returns a list of timestamps (in seconds) where scene changes occur.
    """
    logger.info(f"Detecting scene changes in {video_path} (threshold={threshold})...")

    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-filter:v",
        f"select='gt(scene,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, errors="replace"
        )
    except Exception as e:
        logger.error(f"Failed to run scene detection: {e}")
        return []

    timestamps = []
    # ffmpeg outputs showinfo to stderr
    for line in result.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            match = re.search(r"pts_time:([\d\.]+)", line)
            if match:
                timestamps.append(float(match.group(1)))

    logger.info(f"Detected {len(timestamps)} scene changes.")
    return sorted(timestamps)


def get_scene_density_scores(
    timestamps: list[float],
    duration: float,
    window_size_sec: float = 3.0,
    step_size_sec: float = 1.0,
) -> list[dict]:
    """
    Calculates scene cut density for sliding windows.
    Returns normalized scores (0-100).
    """
    results = []
    current_start = 0.0

    while current_start + window_size_sec <= duration:
        current_end = current_start + window_size_sec
        # Count timestamps within this window
        cuts_in_window = sum(1 for t in timestamps if current_start <= t < current_end)

        results.append({"start": current_start, "end": current_end, "score": float(cuts_in_window)})
        current_start += step_size_sec

    # Normalize
    if results:
        max_cuts = max(r["score"] for r in results)
        if max_cuts > 0:
            for r in results:
                r["score"] = (r["score"] / max_cuts) * 100.0

    return results


def align_and_expand_boundaries(
    start_time: float,
    end_time: float,
    scene_timestamps: list[float],
    min_duration: float = 5.0,
    max_duration: float = 15.0,
) -> tuple[float, float]:
    """
    Expands a core window (e.g. 3 seconds) dynamically to natural scene boundaries.
    Ensures the final clip is between min_duration and max_duration.
    Returns (aligned_start, aligned_end).
    """
    if not scene_timestamps:
        # If no cuts detected, fallback to expanding symmetrically
        pad = (min_duration - (end_time - start_time)) / 2
        return max(0.0, start_time - pad), end_time + pad

    # Find the nearest cut BEFORE or exactly at start_time
    cuts_before = [t for t in scene_timestamps if t <= start_time + 1.0]
    aligned_start = cuts_before[-1] if cuts_before else max(0.0, start_time - 2.0)

    # Find the nearest cut AFTER or exactly at end_time
    cuts_after = [t for t in scene_timestamps if t >= end_time - 1.0]
    aligned_end = cuts_after[0] if cuts_after else end_time + 2.0

    duration = aligned_end - aligned_start

    # If it's too short, keep expanding backward/forward to the next scene cuts
    while duration < min_duration:
        expanded = False

        # Try to expand forward first
        next_cuts_after = [t for t in scene_timestamps if t > aligned_end]
        if next_cuts_after and (next_cuts_after[0] - aligned_start) <= max_duration:
            aligned_end = next_cuts_after[0]
            expanded = True

        # If we can still expand, try backward
        duration = aligned_end - aligned_start
        if duration < min_duration:
            next_cuts_before = [t for t in scene_timestamps if t < aligned_start]
            if next_cuts_before and (aligned_end - next_cuts_before[-1]) <= max_duration:
                aligned_start = next_cuts_before[-1]
                expanded = True

        if not expanded:
            # If we can't find cuts or hit max_duration, break
            break

    # If it happens to be longer than max_duration, we cap it relative to the original core
    duration = aligned_end - aligned_start
    if duration > max_duration:
        # Prefer keeping the core center intact
        core_center = (start_time + end_time) / 2.0
        aligned_start = max(0.0, core_center - (max_duration / 2.0))
        aligned_end = aligned_start + max_duration

    return aligned_start, aligned_end
