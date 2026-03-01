import argparse
import os
import tempfile

from core.scorer import get_highlights
from utils.ffmpeg_helper import export_gif, export_video, extract_audio
from utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Anime Highlight Extractor - Multi-modal Analysis & Export"
    )
    parser.add_argument("video_path", help="Path to the input video file (e.g., .mp4, .mkv)")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top highlights to extract")
    parser.add_argument("--export-mp4", action="store_true", help="Export highlights as MP4 files")
    parser.add_argument(
        "--export-gif", action="store_true", help="Export highlights as high-quality GIF files"
    )
    parser.add_argument(
        "--out-dir", type=str, default=".", help="Directory to save the exported files"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Minimum duration of the extracted clip in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum duration of the extracted clip in seconds",
    )
    parser.add_argument(
        "--weight-audio",
        type=float,
        default=0.5,
        help="Weight for audio energy score (0.0 to 1.0)",
    )
    parser.add_argument(
        "--weight-scene",
        type=float,
        default=0.2,
        help="Weight for scene cut density score (0.0 to 1.0)",
    )
    parser.add_argument(
        "--weight-dynamics",
        type=float,
        default=0.3,
        help="Weight for video dynamics score (0.0 to 1.0)",
    )

    args = parser.parse_args()
    if not os.path.exists(args.video_path):
        logger.error(f"Input video file not found: {args.video_path}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")

        try:
            # 1. Extract audio
            extract_audio(args.video_path, temp_audio_path)

            # 2. Get highlights using the full pipeline
            logger.info("Starting multimodal analysis pipeline...")
            results = get_highlights(
                args.video_path,
                temp_audio_path,
                top_n=args.top_n,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                audio_weight=args.weight_audio,
                scene_weight=args.weight_scene,
                dynamics_weight=args.weight_dynamics,
            )

            # 3. Print the top N segments
            print("\n" + "=" * 70)
            print(f"Top {args.top_n} Highlights for: {base_name}")
            print("=" * 70)

            for i, res in enumerate(results, 1):
                start_sec = res["start"]
                end_sec = res["end"]

                print(
                    f"[{i:02d}] Start: {start_sec:06.2f}s | "
                    f"End: {end_sec:06.2f}s (Duration: {end_sec - start_sec:.2f}s)"
                )
                print(f"     Score: {res['final_score']:05.2f}/100")
                print("-" * 70)

                # Format timestamps for filenames (e.g. 120s to 135s -> 120_135)
                time_suffix = f"{int(start_sec)}_{int(end_sec)}"
                score_suffix = f"s{int(res['final_score'])}"

                file_suffix = f"{time_suffix}_{score_suffix}"

                # 4. Export if requested
                if args.export_mp4:
                    out_path = os.path.join(
                        args.out_dir, f"{base_name}_highlight_{i:02d}_{file_suffix}.mp4"
                    )
                    export_video(args.video_path, start_sec, end_sec, out_path)

                if args.export_gif:
                    out_path = os.path.join(
                        args.out_dir, f"{base_name}_highlight_{i:02d}_{file_suffix}.gif"
                    )
                    export_gif(args.video_path, start_sec, end_sec, out_path)

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}")


if __name__ == "__main__":
    main()
