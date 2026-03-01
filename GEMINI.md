# Anime Highlight Extractor (动漫精彩片段提取器)

## Project Overview
This project is an **Anime Highlight Extractor**, designed to automatically locate and extract the most exciting moments (e.g., intense battles, emotional peaks) from anime videos. It exports these highlights as high-quality MP4s or optimized 256-color GIFs.

**Core Philosophy:** The project relies entirely on **pure non-AI heuristic algorithms**, ensuring extremely fast processing speeds and zero GPU costs. It employs a "Funnel Architecture" (Coarse Filtering -> Fine Verification -> Boundary Alignment -> High-Quality Export).

### Technologies
- **Python**: 3.10+
- **Dependency Management**: `uv`
- **Core Processing Engine**: `ffmpeg-python` (used for audio extraction, scene change detection, and high-quality GIF/MP4 export)
- **Audio Analysis**: `librosa` and `numpy` (for RMS energy extraction)
- **Video Analysis**: `opencv-python` (used only in the fine verification phase for frame difference calculation to minimize overhead)

## Development Conventions
- **Code Formatting & Linting**: Managed by `ruff`.
  - Max line length is set to `100`.
  - Double quotes for strings.
  - Indent with spaces.
- **Pre-commit Hooks**: `pre-commit` is used with `ruff` and `ruff-format` to ensure code is automatically linted and formatted before every commit.
- **Implementation Strategy**: Always adhere to the heuristic design (Audio Energy + Scene Cut Density for Coarse Filtering, OpenCV Frame Differences for Fine Verification). Do not introduce heavy AI dependencies unless specifically requested as a separate fallback layer.

## Building and Running
1. **Virtual Environment & Dependencies**: Managed via `uv`. 
   To install dependencies or add a new package:
   ```bash
   uv add <package_name>
   ```
2. **Pre-commit setup**: Ensure hooks are installed.
   ```bash
   uv run pre-commit install
   ```
3. **Running the Application**: 
   *(TODO: Add exact command once `main.py` is implemented)*
   ```bash
   uv run python main.py <arguments>
   ```

## Directory Overview
- `core/`: Contains the core heuristic analysis modules.
  - `audio_analyzer.py` (Planned): Extracts audio RMS energy.
  - `video_analyzer.py` (Planned): Calculates OpenCV frame differences.
  - `scene_detector.py` (Planned): Identifies FFmpeg scene cuts.
  - `scorer.py` (Planned): Handles the sliding window and composite scoring logic.
- `utils/`: Contains helper modules.
  - `ffmpeg_helper.py` (Planned): Wrappers for FFmpeg extraction and GIF generation.
  - `logger.py` (Planned): Logging utilities.

## Key Files
- `pyproject.toml`: Project configuration and dependency specifications (using `uv`).
- `DESIGN.md`: The architectural blueprint outlining the multi-phase heuristic approach and scoring dimensions (Audio, Dynamics, Scene Density).
- `.pre-commit-config.yaml`: Pre-commit hook configuration running `ruff`.
