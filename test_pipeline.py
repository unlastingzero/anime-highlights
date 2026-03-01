# Mock the heavy dependencies before importing the modules that use them
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Create a dummy cv2 and librosa if they aren't fully available or to speed up tests
mock_cv2 = MagicMock()
mock_librosa = MagicMock()
sys.modules["cv2"] = mock_cv2
sys.modules["librosa"] = mock_librosa
sys.modules["librosa.feature"] = mock_librosa.feature
sys.modules["librosa.onset"] = mock_librosa.onset
sys.modules["librosa.effects"] = mock_librosa.effects

from core.audio_analyzer import analyze_audio_energy  # noqa: E402
from core.scorer import get_highlights  # noqa: E402
from core.video_analyzer import analyze_video_dynamics  # noqa: E402


class TestAnimeHighlightPipeline(unittest.TestCase):
    def setUp(self):
        # Sample data for mocking
        self.dummy_video = "dummy_anime.mp4"
        self.dummy_audio = "dummy_audio.wav"

    @patch("cv2.VideoCapture")
    def test_video_dynamics_logic(self, mock_vc):
        """Tests if video analyzer correctly calculates anime-specific metrics."""
        # Setup mock video capture to return 10 frames
        instance = mock_vc.return_value
        instance.isOpened.return_value = True

        # Mock 10 frames of 100x100 grayscale
        # Frame 5 will be an "impact frame" (bright and high diff)
        frames = []
        for i in range(10):
            frame = np.zeros((100, 100), dtype=np.uint8)
            if i == 5:
                frame.fill(255)  # Bright flash
            frames.append((True, frame))

        # Side effect to return frames then stop
        instance.read.side_effect = frames + [(False, None)]
        instance.get.return_value = 0  # Dummy msec

        # We need to mock cv2.absdiff and cv2.cvtColor
        mock_cv2.cvtColor.side_effect = lambda f, c: f
        mock_cv2.absdiff.side_effect = lambda a, b: np.abs(
            a.astype(float) - b.astype(float)
        ).astype(np.uint8)

        results = analyze_video_dynamics(self.dummy_video, 0, 1)

        self.assertIn("effective_fps", results)
        self.assertIn("impact_score", results)
        # Impact frame at index 5 should trigger high impact_score
        self.assertGreater(results["impact_score"], 0)

    @patch("librosa.load")
    @patch("librosa.get_duration")
    def test_audio_analysis_logic(self, mock_duration, mock_load):
        """Tests if audio analyzer handles advanced features like percussive/noise."""
        mock_duration.return_value = 10.0
        # 10 seconds of silence with 2 hits
        y = np.zeros(1000)
        y[500] = 1.0  # Hit 1
        y[800] = 1.0  # Hit 2
        mock_load.return_value = (y, 100)

        # Mock librosa features
        mock_librosa.feature.rms.return_value = [np.ones(20)]
        mock_librosa.feature.spectral_centroid.return_value = [np.ones(20)]
        mock_librosa.feature.spectral_flatness.return_value = [np.ones(20)]
        mock_librosa.effects.hpss.return_value = (y, y)  # Dummy percussive
        mock_librosa.onset.onset_detect.return_value = [5, 8]
        mock_librosa.frames_to_time.side_effect = lambda f, sr: np.array(f) * 0.1

        results = analyze_audio_energy(self.dummy_audio, window_size_sec=2.0, step_size_sec=1.0)

        self.assertTrue(len(results) > 0)
        self.assertIn("percussive_score", results[0])
        self.assertIn("noise_score", results[0])

    @patch("core.scorer.analyze_audio_energy")
    @patch("core.scorer.detect_scene_changes")
    @patch("core.scorer.analyze_video_dynamics")
    @patch("librosa.get_duration")
    def test_full_pipeline_scoring(self, mock_dur, mock_v_dyn, mock_scenes, mock_audio):
        """Tests the end-to-end scoring and integration of metrics."""
        mock_dur.return_value = 60.0
        mock_scenes.return_value = [10.0, 20.0, 30.0]

        # Mock 10 windows of audio results
        mock_audio.return_value = [
            {
                "start": i,
                "end": i + 3,
                "percussive_score": 10,
                "noise_score": 10,
                "energy_score": 10,
                "onset_score": 10,
                "brightness_score": 10,
            }
            for i in range(10)
        ]
        # Make window at 5.0 high score
        mock_audio.return_value[5]["percussive_score"] = 100

        # Mock video verification for candidates
        mock_v_dyn.return_value = {
            "avg_diff": 50,
            "max_diff": 80,
            "effective_fps": 90,
            "impact_score": 100,
        }

        highlights = get_highlights(self.dummy_video, self.dummy_audio, top_n=1)

        self.assertEqual(len(highlights), 1)
        self.assertIn("final_score", highlights[0])
        self.assertIn("video_metrics", highlights[0])


if __name__ == "__main__":
    unittest.main()
