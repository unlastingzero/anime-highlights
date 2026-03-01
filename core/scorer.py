import librosa

from core.audio_analyzer import analyze_audio_energy
from core.scene_detector import (
    align_and_expand_boundaries,
    detect_scene_changes,
    get_scene_density_scores,
)
from core.video_analyzer import analyze_video_dynamics
from utils.logger import logger


def get_highlights(
    video_path: str,
    temp_audio_path: str,
    top_n: int = 5,
    min_duration: float = 5.0,
    max_duration: float = 30.0,
    audio_energy_weight: float = 0.35,
    audio_onset_weight: float = 0.15,
    scene_weight: float = 0.15,
    dynamics_weight: float = 0.2,
    brightness_weight: float = 0.15,
) -> list[dict]:
    """
    Executes the full heuristic funnel pipeline:
    1. Coarse Filter: Audio Energy + Audio Onset + Scene Cut Density
    2. Fine Verification: OpenCV Frame Dynamics + Brightness Flash
    3. Expansion: Dynamic boundary alignment for continuous shots.
    """
    logger.info("--- PHASE 1: Audio Energy & Onset Analysis ---")
    audio_results = analyze_audio_energy(temp_audio_path)

    if not audio_results:
        logger.warning("No audio results found.")
        return []

    duration = librosa.get_duration(path=temp_audio_path)

    logger.info("--- PHASE 1: Scene Cut Density Analysis ---")
    scene_timestamps = detect_scene_changes(video_path)
    scene_results = get_scene_density_scores(scene_timestamps, duration)

    logger.info("--- Merging Coarse Scores ---")
    merged_coarse = []

    for a_res, s_res in zip(audio_results, scene_results):
        # Weighted Audio Score:
        # We prioritize Percussive (impacts) and Noise (explosions) over raw Energy.
        audio_composite = (
            (a_res["percussive_score"] * 0.4)
            + (a_res["noise_score"] * 0.2)
            + (a_res["energy_score"] * 0.2)
            + (a_res["onset_score"] * 0.1)
            + (a_res["brightness_score"] * 0.1)
        )

        # Merge with Scene Cut Density
        coarse_score = (audio_composite * (audio_energy_weight + audio_onset_weight)) + (
            s_res["score"] * scene_weight
        )

        merged_coarse.append(
            {
                "start": a_res["start"],
                "end": a_res["end"],
                "audio_score": audio_composite,
                "scene_score": s_res["score"],
                "coarse_score": coarse_score,
            }
        )

    merged_coarse.sort(key=lambda x: x["coarse_score"], reverse=True)

    # Merge continuously high-scoring adjacent windows before fine verification
    # If adjacent windows both have high coarse scores, they are part of the same climax.
    threshold_score = merged_coarse[0]["coarse_score"] * 0.6  # Top 40% of the max score

    high_score_blocks = sorted(
        [c for c in merged_coarse if c["coarse_score"] >= threshold_score],
        key=lambda x: x["start"],
    )

    merged_blocks = []
    if high_score_blocks:
        current_block = high_score_blocks[0].copy()

        for block in high_score_blocks[1:]:
            # If windows overlap or are directly adjacent (e.g. step size is 1.0)
            if block["start"] <= current_block["end"] + 1.0:
                current_block["end"] = max(current_block["end"], block["end"])
                # Take average coarse score for the expanded block
                current_block["coarse_score"] = (
                    current_block["coarse_score"] + block["coarse_score"]
                ) / 2.0
            else:
                merged_blocks.append(current_block)
                current_block = block.copy()
        merged_blocks.append(current_block)

    # Sort merged blocks by coarse score
    merged_blocks.sort(key=lambda x: x["coarse_score"], reverse=True)

    candidates = []
    for res in merged_blocks:
        is_overlap = any((res["start"] < c["end"] and res["end"] > c["start"]) for c in candidates)
        if not is_overlap:
            candidates.append(res)
        if len(candidates) >= top_n * 2:
            break

    msg = f"--- PHASE 2: Fine Verification ({len(candidates)} candidates) ---"
    logger.info(msg)

    for cand in candidates:
        v_res = analyze_video_dynamics(video_path, cand["start"], cand["end"])
        cand["video_metrics"] = v_res

    # Normalize metrics across candidates
    metrics_to_normalize = ["avg_diff", "max_diff", "effective_fps", "impact_score"]
    max_vals = {}
    for m in metrics_to_normalize:
        max_val = max((c["video_metrics"][m] for c in candidates), default=0.0)
        max_vals[m] = max_val if max_val > 0 else 1.0

    for cand in candidates:
        v = cand["video_metrics"]
        # Normalize each sub-metric to 0-100
        norm_avg = (v["avg_diff"] / max_vals["avg_diff"]) * 100.0
        norm_max = (v["max_diff"] / max_vals["max_diff"]) * 100.0
        norm_fps = (v["effective_fps"] / max_vals["effective_fps"]) * 100.0
        norm_impact = (v["impact_score"] / max_vals["impact_score"]) * 100.0

        # Composite video dynamics score (Weighted)
        # We give more weight to effective_fps (Sakuga) and impact_score
        video_score = (norm_avg * 0.2) + (norm_max * 0.2) + (norm_fps * 0.4) + (norm_impact * 0.2)

        cand["video_score"] = video_score
        # Combine coarse score with the new refined video score
        # Using the sum of previous dynamics_weight and brightness_weight for the combined video_score # noqa: E501
        video_weight_total = dynamics_weight + brightness_weight
        cand["final_score"] = cand["coarse_score"] + (video_score * video_weight_total)

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    final_candidates = candidates[:top_n]

    logger.info("--- PHASE 3: Dynamic Scene Boundary Alignment & Final Deduplication ---")

    for cand in final_candidates:
        orig_start, orig_end = cand["start"], cand["end"]
        # Expand based on actual scene cuts to make it a continuous shot
        aligned_start, aligned_end = align_and_expand_boundaries(
            orig_start,
            orig_end,
            scene_timestamps,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        cand["start"] = aligned_start
        cand["end"] = aligned_end
        logger.info(
            f"Expanded [{orig_start:.2f}-{orig_end:.2f}] -> [{aligned_start:.2f}-{aligned_end:.2f}]"
        )

    # Deduplicate again after expansion to ensure no overlapping clips are returned
    # Overlap logic: If two clips share more than 30% of their length, keep the higher scoring one.
    unique_candidates = []
    for cand in final_candidates:
        is_duplicate = False
        for u_cand in unique_candidates:
            # Calculate overlap duration
            overlap_start = max(cand["start"], u_cand["start"])
            overlap_end = min(cand["end"], u_cand["end"])
            overlap_duration = max(0, overlap_end - overlap_start)

            # If they overlap significantly, treat as duplicate
            if overlap_duration > 0:
                min_len = min(cand["end"] - cand["start"], u_cand["end"] - u_cand["start"])
                if (overlap_duration / min_len) > 0.3:  # 30% overlap threshold
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_candidates.append(cand)

    logger.info(f"--- Analysis Complete. Returning {len(unique_candidates)} unique highlights ---")
    return unique_candidates
