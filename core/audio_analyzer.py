import numpy as np

from utils.logger import logger


def analyze_audio_energy(
    audio_path: str, window_size_sec: float = 3.0, step_size_sec: float = 1.0
) -> list[dict]:
    """
    Analyzes audio with Anime-specific acoustic heuristics:
    - energy_score (RMS): Overall volume.
    - onset_score: Density of sound hits.
    - percussive_score: Strength of percussive/impact sounds (explosions, hits).
    - brightness_score (Spectral Centroid): High-frequency intensity (clashes, beams).
    - noise_score (Spectral Flatness): Detection of "explosive" white-noise like sounds.
    """
    import librosa

    logger.info(f"Analyzing advanced audio features from {audio_path}...")

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # 1. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)

    # 2. Percussive component (HPSS) - Great for isolating hits/explosions
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    percussive_rms = librosa.feature.rms(y=y_percussive)[0]

    # 3. Spectral Centroid (Brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # 4. Spectral Flatness (Noise-like/Explosive quality)
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # 5. Onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    results = []
    current_start = 0.0

    while current_start + window_size_sec <= duration:
        current_end = current_start + window_size_sec

        idx_start = np.searchsorted(times, current_start)
        idx_end = np.searchsorted(times, current_end)

        if idx_start < idx_end:
            # Aggregate features for the window (using 90th percentile for robustness)
            win_rms = np.percentile(rms[idx_start:idx_end], 90)
            win_percussive = np.percentile(percussive_rms[idx_start:idx_end], 90)
            win_centroid = np.percentile(centroid[idx_start:idx_end], 90)
            win_flatness = np.percentile(flatness[idx_start:idx_end], 90)
        else:
            win_rms = win_percussive = win_centroid = win_flatness = 0.0

        # Onset count
        o_start = np.searchsorted(onset_times, current_start)
        o_end = np.searchsorted(onset_times, current_end)
        onset_count = o_end - o_start

        results.append(
            {
                "start": current_start,
                "end": current_end,
                "energy_score": float(win_rms),
                "percussive_score": float(win_percussive),
                "brightness_score": float(win_centroid),
                "noise_score": float(win_flatness),
                "onset_score": float(onset_count),
            }
        )
        current_start += step_size_sec

    # Normalize all scores to 0-100
    metrics = ["energy_score", "percussive_score", "brightness_score", "noise_score", "onset_score"]
    for m in metrics:
        max_val = max((r[m] for r in results), default=0.0)
        if max_val > 0:
            for r in results:
                r[m] = (r[m] / max_val) * 100.0

    logger.info(f"Completed advanced audio analysis. Windows: {len(results)}")
    return results
