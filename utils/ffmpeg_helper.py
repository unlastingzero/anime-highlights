import os
import sys

import ffmpeg

from utils.logger import logger


def get_ffmpeg_path():
    """
    Returns the path to the ffmpeg executable.
    When frozen with PyInstaller, it checks for a bundled binary first.
    If not found, it falls back to the system 'ffmpeg' command.
    """
    if getattr(sys, "frozen", False):
        # Look for bundled ffmpeg in the temp directory
        bundled_path = os.path.join(sys._MEIPASS, "ffmpeg")
        if os.path.exists(bundled_path):
            return bundled_path

    # Default to system ffmpeg (requires user to have it installed)
    return "ffmpeg"


def extract_audio(video_path: str, output_audio_path: str, sample_rate: int = 16000) -> str:
    """
    Extracts audio from a video file and saves it as a WAV file.
    Downsamples to a single mono channel to significantly speed up processing.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Extracting audio from {video_path} (Mono, {sample_rate}Hz)...")
    ffmpeg_cmd = get_ffmpeg_path()

    try:
        (
            ffmpeg.input(video_path)
            .output(output_audio_path, ac=1, ar=sample_rate, loglevel="error")
            .overwrite_output()
            .run(cmd=ffmpeg_cmd, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode("utf8")
        logger.error(f"FFmpeg failed to extract audio: {error_msg}")
        raise RuntimeError(f"FFmpeg failed to extract audio: {error_msg}") from e

    logger.info("Audio extraction complete.")
    return output_audio_path


def export_video(video_path: str, start_time: float, end_time: float, output_path: str) -> None:
    """
    Cuts a segment of the video and exports it as an MP4 without re-encoding (if possible).
    """
    logger.info(f"Exporting MP4: [{start_time:.2f}s - {end_time:.2f}s] to {output_path}")
    duration = end_time - start_time
    ffmpeg_cmd = get_ffmpeg_path()

    try:
        (
            ffmpeg.input(video_path, ss=start_time, t=duration)
            .output(output_path, c="copy", loglevel="error")
            .overwrite_output()
            .run(cmd=ffmpeg_cmd, capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"Successfully exported {output_path}")
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode("utf8")
        logger.error(f"FFmpeg failed to export video: {error_msg}")


def export_gif(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    width: int = 240,
    fps: int = 15,
) -> None:
    """
    Cuts a segment of the video and exports it as a high-quality GIF using a custom palette.
    """
    logger.info(f"Exporting GIF (HQ): [{start_time:.2f}s - {end_time:.2f}s] to {output_path}")
    duration = end_time - start_time
    ffmpeg_cmd = get_ffmpeg_path()

    # We use complex filter to generate a custom palette for the exact segment,
    # then map it to get a high-quality GIF without dithering artifacts.
    try:
        stream = ffmpeg.input(video_path, ss=start_time, t=duration)

        # Scale and adjust framerate
        video = stream.video.filter("fps", fps=fps).filter("scale", width, -1, flags="lanczos")

        # Split the video stream to use it for both palette generation and final mapping
        split = video.split()

        # Generate palette
        palette = split[0].filter("palettegen")

        # Use palette to generate gif
        (
            ffmpeg.filter([split[1], palette], "paletteuse")
            .output(output_path, loop=0, loglevel="error")
            .overwrite_output()
            .run(cmd=ffmpeg_cmd, capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"Successfully exported {output_path}")
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode("utf8")
        logger.error(f"FFmpeg failed to export GIF: {error_msg}")
