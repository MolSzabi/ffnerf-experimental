"""Source-level guards for training bugs that do not require an OpenCL runtime."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TrainingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = (ROOT / "ffnerf_tools.py").read_text(encoding="utf-8")
        cls.kernel = (ROOT / "ffnerf_OG.cl").read_text(encoding="utf-8")

    def test_sigma_bias_targets_density_output(self):
        self.assertIn('if bn == "bd_out":', self.host)
        self.assertNotIn('if bn == "bd_1":\n                    if b_f.size > 0:\n                        b_f[0] = SIGMA_BIAS', self.host)

    def test_occupancy_maintenance_is_reachable_and_full_grid(self):
        self.assertIn('self.steps_trained % OCC_MAINTENANCE_PERIOD == 0', self.host)
        self.assertNotIn('self.steps_trained % (32)+1', self.host)
        self.assertIn('and float(occ_update_strength) > 0.0', self.host)
        self.assertIn('n_words = int(self.occ.G_occ) ** 3', self.host)
        self.assertIn('global_ = ((n_words + local - 1) // local) * local', self.host)
        self.assertIn('evt_decay = cl.enqueue_nd_range_kernel(', self.host)

    def test_core_maintenance_does_not_enter_suppress_kernel(self):
        start = self.host.index('do_occ_maintenance = (')
        end = self.host.index('ev_ff = cl.enqueue_nd_range_kernel(', start)
        launch_selection = self.host[start:end]
        self.assertIn('ffnerf_kernel_to_run = self.ffnerf_knl', launch_selection)
        self.assertNotIn('ffnerf_kernel_to_run = self.ffnerf_suppress_mode_knl', launch_selection)

    def test_background_define_matches_host(self):
        self.assertIn('#ifdef BACKGROUND_COLOR', self.kernel)
        self.assertIn('#if (BACKGROUND_COLOR==0)', self.kernel)
        self.assertNotIn('#ifdef BACKGROUND\n', self.kernel)

    def test_occupancy_rounding_and_keep_paths(self):
        self.assertIn('uint q_stoch = (r24 < thresh) ? (q0 + 1u) : q0;', self.kernel)
        self.assertIn('if (keep_p >= 1.0f)', self.kernel)
        self.assertIn('if (rand01(&rng_keep) <= keep_p)', self.kernel)


if __name__ == '__main__':
    unittest.main()
