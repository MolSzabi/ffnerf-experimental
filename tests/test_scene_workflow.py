"""CPU checks for the SfM contract; no GPU or real dataset required."""
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from scene_config import load_config
from sfm_data import read_scene, load_images, scaled_intrinsics, scene_bounds


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for i in range(2):
            cv2.imwrite(str(self.root / f'{i}.png'), np.full((80, 40, 3), 80 + i, np.uint8))
        self.K = np.array([[40, 0, 10], [0, 40, 20], [0, 0, 1]], np.float32)
        self.poses = np.tile(np.eye(4), (2, 1, 1))
        self.poses[1, :3, 3] = [1, 2, 3]
        self.data = {'reconstruction': str(self.root / 'scene.npz'), 'images_dir': str(self.root),
                     'sfm_max_side': 40, 'max_side': 20}

    def save(self, **kwargs):
        np.savez(self.data['reconstruction'], K=self.K,
                 img_paths=np.array(['old/place/0.png', 'old/place/1.png'], dtype=object), **kwargs)

    def test_ba_w2c_and_relocated_images(self):
        self.save(RT=np.linalg.inv(self.poses), rt_dir=np.array(['w2c']), active_cam_global=[8, 12])
        paths, poses, K, ids, points = read_scene(self.data)
        np.testing.assert_allclose(poses, self.poses)
        self.assertEqual(ids.tolist(), [8, 12])
        images, K, wh = load_images(self.data, paths, K)
        self.assertEqual(images.shape, (2, 20, 10, 3))
        np.testing.assert_allclose(K[:2], self.K[:2] / 2)

    def test_stage2_filters_before_loading_images(self):
        self.save(c2w=self.poses, posed=[True, True], keep_mask=[False, True],
                  X_track=[[1, 2, 3], [9, 9, 9]], has_X=[True, False])
        self.data['images_dir'] = str(self.root / 'selected')
        Path(self.data['images_dir']).mkdir()
        cv2.imwrite(str(Path(self.data['images_dir']) / '1.png'), np.zeros((80, 40, 3), np.uint8))
        paths, poses, K, ids, points = read_scene(self.data)
        self.assertEqual(ids.tolist(), [1])
        np.testing.assert_allclose(points, [[1, 2, 3]])
        self.assertEqual(len(paths), 1)

    def test_geometry_and_bounds(self):
        np.testing.assert_allclose(scaled_intrinsics(self.K, (40, 80), (20, 40), 40), self.K)
        with self.assertRaises(ValueError):
            scene_bounds({'box_min': [1, 0, 0], 'box_max': [0, 1, 1]}, np.empty((0, 3)))
        lo, hi = scene_bounds({'box_min': None, 'box_max': None}, np.array([[1, 2, 3]]))
        self.assertTrue(np.all(hi > lo))

        points = np.array([[0, 0, 0], [2, 4, 6], [1, 2, 3]], np.float32)
        q_lo, q_hi = np.percentile(points, [1, 99], axis=0)
        center = 0.5 * (q_lo + q_hi)
        radius = max(0.5 * np.linalg.norm(q_hi - q_lo), 0.01) * 1.5
        lo, hi = scene_bounds({'box_min': None, 'box_max': None}, points)
        np.testing.assert_allclose((lo + hi) * 0.5, center, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose((hi - lo) * 0.5, np.full(3, radius), rtol=1e-6, atol=1e-6)

    def test_review_rays_match_training_intrinsics(self):
        import ffnerf_tools as ft
        images = np.zeros((2, 80, 40, 3), np.float32)
        u, v = np.array([0, 19, 39]), np.array([0, 29, 79])
        training, _ = ft.rays_targets_from_indices(np.ones(3, np.int32), u, v,
                                                  images, self.poses, 40, 80, K=self.K)
        review = ft.rays_from_rt_indices(self.poses[1], u, v, 40, 80, K=self.K)
        np.testing.assert_allclose(training['o'], review['o'])
        np.testing.assert_allclose(training['d'], review['d'], atol=1e-7)

    def test_config_paths_and_unknown_keys(self):
        path = self.root / 'scene.json'
        path.write_text(json.dumps({'data': {'reconstruction': 'scene.npz'}}))
        self.assertEqual(load_config(path)['data']['reconstruction'], str(self.root / 'scene.npz'))
        path.write_text(json.dumps({'training': {'stepz': 10}}))
        with self.assertRaises(ValueError):
            load_config(path)


if __name__ == '__main__':
    unittest.main()
