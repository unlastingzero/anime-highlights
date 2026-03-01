# Anime Highlight Extractor 🎬✨

Anime Highlight Extractor 是一个专注于**动漫视频**的“精彩高能片段”自动提取工具。

针对动漫的特殊表现形式（热血 BGM、大声嘶吼、高频分镜、强烈的特效闪烁），本项目采用了一套**纯启发式（Heuristic-based）**的多模态分析算法。它能在**完全不需要 GPU、不需要调用任何昂贵 AI 大模型**的情况下，以极快的速度（一集 24 分钟的动漫只需几十秒）扫描视频，自动锁定最精彩的决战或情绪爆发点，并将其无损截取为 MP4 或高质量的动漫专属优化 GIF。

---

## 🌟 核心特性

本项目采用独特的 **“粗筛 -> 精筛 -> 动态镜头对齐”** 漏斗式架构，保证了处理速度与精确度的完美平衡：

1. **音频能量分析 (粗筛 35%)**：极速剥离音轨，使用滑动窗口和 90 分位 RMS 能量算法，精准锁定高燃 BGM 或角色怒吼的片段。
2. **音频突变密度 (粗筛 15%)**：利用 librosa 提取 onset，精准捕捉密集的打击声、爆炸声或快节奏鼓点。
3. **镜头切分密度检测 (粗筛 15%)**：精彩战斗往往伴随极快的镜头切换。通过底层 FFmpeg 解析视频硬切分点，过滤掉长镜头的“伪高潮”（如高音量的静态片尾曲）。
4. **视觉运动强度计算 (精筛 20%)**：利用 OpenCV 对粗筛出的核心区间进行像素级的帧差分析，捕捉剧烈的物理运动。
5. **画面亮度跳变率 (精筛 15%)**：通过计算每一帧平均亮度的方差，专门捕捉动漫表现剧烈力量碰撞、大招对轰时特有的“冲击帧” (Impact Frames) 或光污染特效。
6. **动态语义镜头对齐**：系统不会生硬地按固定秒数（如 3 秒）切断视频。它会以高潮点为中心，向前向后寻找最近的“画面硬切分点”，将其动态扩展成一个 **5~15秒的完整长镜头**。
7. **高质量导出**：
   - **MP4**：秒速无损流拷贝（Zero Re-encoding）。
   - **GIF**：使用 FFmpeg 复杂的 `palettegen/paletteuse` 滤镜，为每一段高燃动画生成专属的 256 色色板，彻底解决传统 GIF 的颗粒感。

---

## 🛠️ 安装指南

本项目使用 `uv` 进行现代化的 Python 依赖管理，并依赖系统级的 `ffmpeg` 工具。

### 1. 安装系统依赖 (FFmpeg)

必须在你的操作系统上安装 `ffmpeg`：

- **macOS** (使用 Homebrew):
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt install ffmpeg
  ```

### 2. 初始化 Python 环境并安装依赖

确保你已经安装了 [uv](https://github.com/astral-sh/uv)，然后在项目根目录下运行：

```bash
# 同步并安装所有必要的 Python 库 (librosa, opencv-python, ffmpeg-python 等)
uv sync
```

_(可选) 安装开发环境依赖 (用于代码格式化与校验)_：

```bash
uv run pre-commit install
```

---

## 🚀 使用方法

通过命令行运行 `main.py` 即可一键分析并提取视频片段。

### 基本使用 (仅打印分析结果)

默认会输出排名前 3 的高燃片段的时间戳和各项得分：

```bash
uv run python main.py "/path/to/your/anime.mp4"
```

### 自动导出视频与 GIF

你可以使用 `--export-mp4` 和 `--export-gif` 参数让工具自动将分析出的高潮片段剪辑并保存在指定的目录下。

```bash
uv run python main.py "/path/to/your/anime.mp4" \
    --top-n 1 \
    --export-mp4 \
    --export-gif \
    --out-dir ./outputs \
    --max-duration 60 \
    --weight-audio-energy 0.2 \
    --weight-audio-onset 0.2 \
    --weight-scene 0.2 \
    --weight-dynamics 0.2 \
    --weight-brightness 0.2 \
```

### 常用参数说明

| 参数                    | 说明                                                   | 默认值         |
| :---------------------- | :----------------------------------------------------- | :------------- |
| `video_path`            | (必填) 输入的动漫视频文件路径。                        | -              |
| `--top-n`               | 要提取并显示的前 N 个精彩片段数量。                    | `3`            |
| `--export-mp4`          | 添加此标志后，自动将精彩片段无损截取为 `.mp4` 文件。   | `False`        |
| `--export-gif`          | 添加此标志后，自动将精彩片段生成为高质量 `.gif` 文件。 | `False`        |
| `--out-dir`             | 导出文件的存放目录。如果目录不存在会自动创建。         | `.` (当前目录) |
| `--min-duration`        | 截取视频片段的最短时间（秒）。                         | `5.0`          |
| `--max-duration`        | 截取视频片段的最长时间（秒）。                         | `30.0`         |
| `--weight-audio-energy` | 音频能量在综合打分中的权重比例。                       | `0.35`         |
| `--weight-audio-onset`  | 音频突变密度在综合打分中的权重比例。                   | `0.15`         |
| `--weight-scene`        | 镜头切分密度在综合打分中的权重比例。                   | `0.15`         |
| `--weight-dynamics`     | 画面运动强度在综合打分中的权重比例。                   | `0.20`         |
| `--weight-brightness`   | 画面亮度跳变率在综合打分中的权重比例。                 | `0.15`         |

### 导出文件命名规则

当你使用 `--export-mp4` 或 `--export-gif` 时，输出的文件名将会包含原视频名称、排名、具体的起始和结束时间（以秒为单位）以及该片段的综合得分（前缀为 `s`），例如：
`st_hero_08_highlight_01_1277_1286_s88.mp4`
表示该片段是第一高能时刻，截取自原视频的 1277 秒至 1286 秒，其综合评分为 88 分。

---

## ⚖️ 打分权重与推荐配置

系统基于五个核心维度进行综合打分：**音频能量**、**音频突变密度**、**镜头切分密度**、**画面运动强度** 和 **画面亮度跳变率**。默认的权重配置最适合典型的热血战斗番。

针对不同类型的动画，你可以通过命令行参数灵活调整这些权重（建议权重总和为 `1.0`）：

#### 1. 典型热血/战斗番 (默认配置)

表现最为均衡，能够完美捕捉“大吼/打击音效 + 冲击帧/高频分镜”。

```bash
--weight-audio-energy 0.35 --weight-audio-onset 0.15 --weight-scene 0.15 --weight-dynamics 0.20 --weight-brightness 0.15
```

#### 2. 日常、催泪或剧情向动画

这类动画高光时刻往往是角色安静的流泪或深情的独白，可能声音不大也没有冲击帧。应侧重画面本身的张力和情感转折。

```bash
--weight-audio-energy 0.30 --weight-audio-onset 0.10 --weight-scene 0.20 --weight-dynamics 0.40 --weight-brightness 0.0
```

#### 3. 早期老番或“PPT”动画

画面经常是一张静态图在平移，主要依靠配乐或密集的打击音效来表现燃点。

```bash
--weight-audio-energy 0.40 --weight-audio-onset 0.30 --weight-scene 0.20 --weight-dynamics 0.05 --weight-brightness 0.05
```

#### 4. AMV / MAD (动漫音乐混剪)

整首歌从头到尾都很响（能量失效），此时完全只能依靠密集鼓点和华丽转场来寻找。

```bash
--weight-audio-energy 0.05 --weight-audio-onset 0.25 --weight-scene 0.40 --weight-dynamics 0.15 --weight-brightness 0.15
```

---

## 🏗️ 进阶配置与微调

如果你希望进一步微调，可以修改代码中的以下参数：

- **GIF 分辨率与帧率** (`utils/ffmpeg_helper.py` 中的 `export_gif`)：
  默认为宽度 `480px`，帧率 `15fps` 以平衡体积与流畅度。如需超清 GIF，可调整为 `width=720, fps=24`。
- **打分权重** (`core/scorer.py` 中的 `get_highlights`)：
  可在代码中直接调整默认的 5 维权重分布。
