import tempfile
import unittest
from pathlib import Path

from scripts import visualize_attention_roi as visualization


class AttentionVisualizationPathTest(unittest.TestCase):
    def test_colab_absolute_video_path_is_remapped_to_project_data(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "local-project"
            local_video = project_root / "data" / "example" / "video.mp4"
            local_video.parent.mkdir(parents=True)
            local_video.touch()

            resolved = visualization.resolve_video_path(
                "/content/vlm-event-boundary/data/example/video.mp4",
                project_root,
            )

        self.assertEqual(resolved, local_video)


if __name__ == "__main__":
    unittest.main()
