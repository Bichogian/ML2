from unittest import TestCase

from btc15m.protocolo import utc_ns
from btc15m.walkforward import demo_history, slice_sample


class TestWalkForward(TestCase):
    def test_cut_rejects_crossing_windows_and_unavailable_labels(self):
        history = demo_history()
        start, end = "2026-04-01", "2026-05-01"
        sample = slice_sample(history, start, end, settled_before_end=True)
        rejected_crossing, rejected_label = sample.frame["ticker"].iloc[:2]
        history.frame.loc[history.frame["ticker"].eq(rejected_crossing), "btc_first_close_ns"] = utc_ns(start) - 1
        history.frame.loc[history.frame["ticker"].eq(rejected_label), "settled_ns"] = utc_ns(end)
        adjusted = slice_sample(history, start, end, settled_before_end=True)
        self.assertEqual(len(adjusted.frame), len(sample.frame) - 2)
        self.assertTrue((adjusted.frame["btc_first_close_ns"] >= utc_ns(start)).all())
        self.assertTrue((adjusted.frame["settled_ns"] < utc_ns(end)).all())
