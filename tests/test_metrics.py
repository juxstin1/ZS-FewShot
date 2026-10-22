import unittest

from zsbench.metrics import bootstrap_ci, expected_calibration_error, percentile, summarize
from zsbench.task import near_copies

LABELS = ["a", "b", "c"]


def row(gold, pred, confidence=None, latency_ms=10.0, **extra):
    r = {"gold": gold, "pred": pred, "confidence": confidence, "latency_ms": latency_ms}
    if extra:
        r["extra"] = extra
    return r


class MetricsTest(unittest.TestCase):
    def test_accuracy_f1_invalid(self):
        rows = [row("a", "a"), row("a", "b"), row("b", "b"), row("c", None)]
        s = summarize(rows, LABELS)
        self.assertEqual(s["accuracy"], 0.5)
        self.assertEqual(s["invalid"], 1)
        # a: P=1 R=.5 F1=.667; b: P=.5 R=1 F1=.667; c: F1=0 -> macro over a,b,c
        self.assertAlmostEqual(s["macro_f1"], (2 / 3 + 2 / 3 + 0) / 3)
        self.assertEqual(s["confusion"]["c"], {"<invalid>": 1})

    def test_macro_f1_skips_labels_never_seen(self):
        s = summarize([row("a", "a"), row("b", "b")], LABELS)
        self.assertEqual(s["macro_f1"], 1.0)

    def test_percentile_nearest_rank(self):
        values = [float(v) for v in range(1, 101)]
        self.assertEqual(percentile(values, 50), 50.0)
        self.assertEqual(percentile(values, 95), 95.0)
        self.assertIsNone(percentile([], 50))

    def test_ece(self):
        # Two rows at 0.9 confidence, one right one wrong: |0.5 - 0.9| = 0.4.
        rows = [row("a", "a", 0.9), row("a", "b", 0.9)]
        self.assertAlmostEqual(expected_calibration_error(rows), 0.4)
        self.assertIsNone(expected_calibration_error([row("a", "a")]))

    def test_two_way_rows_accept_either_label(self):
        rows = [
            {**row("a", "b"), "accepted": ["a", "b"]},  # picked the second accepted label: correct
            {**row("a", "c"), "accepted": ["a", "b"]},  # picked neither: wrong, scored against "a"
        ]
        s = summarize(rows, LABELS)
        self.assertEqual(s["accuracy"], 0.5)
        self.assertEqual(s["ambiguous"], 2)
        self.assertEqual(s["confusion"], {"b": {"b": 1}, "a": {"c": 1}})

    def test_bootstrap_ci_brackets_accuracy(self):
        hits = [True] * 75 + [False] * 25
        lo, hi = bootstrap_ci(hits)
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)
        self.assertLess(hi - lo, 0.25)
        self.assertEqual(bootstrap_ci([True] * 10), (1.0, 1.0))
        self.assertIsNone(bootstrap_ci([]))

    def test_order_flips_and_first_order_accuracy(self):
        rows = [
            row("a", "a", 0.5, order_choices=["a", "a"], first_order="a"),
            row("b", "b", 0.5, order_choices=["a", "b"], first_order="a"),
        ]
        s = summarize(rows, LABELS)
        self.assertEqual(s["flip_rate"], 0.5)
        self.assertEqual(s["first_order_accuracy"], 0.5)


class NearCopiesTest(unittest.TestCase):
    def test_case_punctuation_and_small_rewordings(self):
        texts = [
            "Fix it.",
            "fix it!",
            "Run the test suite and tell me which tests fail",
            "Please run the test suite and tell me which tests fail",
            "Run the linter",
        ]
        self.assertEqual(near_copies(texts), [(0, 1), (2, 3)])

    def test_short_texts_need_an_exact_match(self):
        # 3-word texts never match on word overlap alone.
        self.assertEqual(near_copies(["fix the bug"], ["fix the bugs"]), [])
        self.assertEqual(near_copies(["Fix the bug"], ["fix the BUG."]), [(0, 0)])


if __name__ == "__main__":
    unittest.main()
