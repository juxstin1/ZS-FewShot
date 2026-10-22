import importlib.util
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "gpu_guard", Path(__file__).resolve().parent.parent / "scripts" / "gpu_guard.py")
gpu_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_guard)


def state(used_gb, free_gb, util=10):
    return {"name": "GPU", "total": 24 * 1024, "used": int(used_gb * 1024), "free": int(free_gb * 1024), "util": util}


class PreflightTest(unittest.TestCase):
    def test_room_left_for_prod(self):
        with mock.patch.object(gpu_guard, "gpu_state", return_value=state(5, 18)):
            ok, report = gpu_guard.preflight(need_gb=11, reserve_gb=4, max_util=60)
        self.assertTrue(ok)
        self.assertIn("7.0 GB free", report)

    def test_refuses_when_reserve_would_be_eaten(self):
        with mock.patch.object(gpu_guard, "gpu_state", return_value=state(10, 13)):
            ok, report = gpu_guard.preflight(need_gb=11, reserve_gb=4, max_util=60)
        self.assertFalse(ok)
        self.assertIn("not enough VRAM", report)

    def test_refuses_when_gpu_is_busy(self):
        with mock.patch.object(gpu_guard, "gpu_state", return_value=state(5, 18, util=85)):
            ok, report = gpu_guard.preflight(need_gb=11, reserve_gb=4, max_util=60)
        self.assertFalse(ok)
        self.assertIn("busy", report)


if __name__ == "__main__":
    unittest.main()
