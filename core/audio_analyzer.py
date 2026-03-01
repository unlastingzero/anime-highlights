import librosa
import numpy as np

from utils.logger import logger


def analyze_audio_energy(
    audio_path: str, window_size_sec: float = 3.0, step_size_sec: float = 1.0
) -> list[dict]:
    """
    Analyzes the audio file and calculates the RMS energy for sliding windows.
    Returns normalized scores between 0 and 100.
    """
    logger.info(f"Analyzing audio energy from {audio_path}...")

    # Load audio. sr=None preserves the sample rate we set during extraction (e.g., 16000Hz)
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    logger.info(f"Audio loaded. Duration: {duration:.2f}s, Sample Rate: {sr}Hz")

    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)

    results = []
    current_start = 0.0

    # Sliding window logic
    while current_start + window_size_sec <= duration:
        current_end = current_start + window_size_sec

        start_idx = np.searchsorted(times, current_start)
        end_idx = np.searchsorted(times, current_end)

        if start_idx < end_idx:
            window_rms = rms[start_idx:end_idx]
            # Use 90th percentile to represent the "high energy" of the window,
            # which is more robust against short, single static pops than taking the max().
            score = np.percentile(window_rms, 90)
        else:
            score = 0.0

        results.append(
            {
                "start": current_start,
                "end": current_end,
                "score": float(score),
            }
        )
        current_start += step_size_sec

    # Normalize scores to 0-100 scale for easier weighting with other features
    if results:
        max_score = max(r["score"] for r in results)
        if max_score > 0:
            for r in results:
                r["score"] = (r["score"] / max_score) * 100.0

    logger.info(f"Completed energy analysis. Generated {len(results)} windows.")
    return results
