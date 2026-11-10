"""Negative tests ensure bad or incomplete reports cannot receive PASS."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_memprof import EXPECTED, validate_report


def valid_report(case="interleaved"):
    peak, count = EXPECTED[case]
    return {
        "format_version": 1, "complete": True, "num_devices": 1,
        "peak_total_bytes": peak, "current_total_bytes": 0, "live_allocations": 0,
        "devices": [{"device": 0, "peak_bytes": peak, "current_bytes": 0,
                     "managed_peak_bytes": 0, "managed_current_bytes": 0,
                     "allocs": count, "frees": count, "failed_allocs": 0,
                     "largest_failed_bytes": 0}],
        "host_pinned": {"peak_bytes": 0, "current_bytes": 0},
        "incomplete_reasons": [], "incomplete_reasons_dropped": 0,
        "anomalies": {"unknown_frees": 0, "invalid_args": 0,
                      "duplicate_allocs": 0, "null_allocs": 0, "internal_errors": 0},
    }


class CheckerTests(unittest.TestCase):
    def test_valid_cases(self):
        for case in EXPECTED:
            with self.subTest(case=case):
                validate_report(case, valid_report(case))

    def test_reject_wrong_or_missing_root_fields(self):
        changes = {"peak_total_bytes": 1, "current_total_bytes": 1,
                   "live_allocations": 1, "complete": False, "format_version": 2,
                   "incomplete_reasons": [{"reason": "missing API", "count": 1}],
                   "incomplete_reasons_dropped": 1, "num_devices": 0}
        for key, value in changes.items():
            with self.subTest(key=key, failure="wrong"):
                report = valid_report()
                report[key] = value
                with self.assertRaises(ValueError):
                    validate_report("interleaved", report)
            with self.subTest(key=key, failure="missing"):
                report = valid_report()
                del report[key]
                with self.assertRaises(ValueError):
                    validate_report("interleaved", report)

    def test_reject_wrong_device_accounting(self):
        for key in ("peak_bytes", "current_bytes", "allocs", "frees", "failed_allocs",
                    "largest_failed_bytes", "managed_peak_bytes", "managed_current_bytes"):
            with self.subTest(key=key):
                report = valid_report()
                report["devices"][0][key] += 1
                with self.assertRaises(ValueError):
                    validate_report("interleaved", report)

    def test_reject_anomalies(self):
        for key in (*valid_report()["anomalies"], "future_anomaly"):
            with self.subTest(key=key):
                report = valid_report()
                report["anomalies"][key] = 1
                with self.assertRaises(ValueError):
                    validate_report("interleaved", report)

    def test_reject_missing_anomaly_counter(self):
        report = valid_report()
        del report["anomalies"]["unknown_frees"]
        with self.assertRaises(ValueError):
            validate_report("interleaved", report)

    def test_reject_wrong_types(self):
        for key, value in (("complete", 1), ("current_total_bytes", False),
                           ("peak_total_bytes", "234881024"), ("devices", {})):
            with self.subTest(key=key):
                report = valid_report()
                report[key] = value
                with self.assertRaises(ValueError):
                    validate_report("interleaved", report)
        for report in (None, [], 42):
            with self.assertRaises(ValueError):
                validate_report("interleaved", report)

    def test_idle_extra_device_allowed(self):
        report = valid_report()
        idle = {key: 0 for key in report["devices"][0]}
        idle["device"] = 1
        report["devices"].append(idle)
        report["num_devices"] = 2
        validate_report("interleaved", report)
        report["devices"][1]["peak_bytes"] = 1
        with self.assertRaises(ValueError):
            validate_report("interleaved", report)

    def test_reject_duplicate_device_ids(self):
        report = valid_report()
        report["num_devices"] = 2
        report["devices"].append(copy.deepcopy(report["devices"][0]))
        with self.assertRaises(ValueError):
            validate_report("interleaved", report)

    def test_reject_pinned_memory(self):
        report = valid_report()
        report["host_pinned"]["peak_bytes"] = 1
        with self.assertRaises(ValueError):
            validate_report("interleaved", report)

    def test_cli_exit_status(self):
        script = Path(__file__).resolve().parents[1] / "check_memprof.py"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            command = [sys.executable, str(script), "interleaved", str(path)]
            path.write_text(json.dumps(valid_report()), encoding="utf-8")
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            for bad in ("not JSON", "{}", '{"complete": false}'):
                path.write_text(bad, encoding="utf-8")
                self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            path.unlink()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)


if __name__ == "__main__":
    unittest.main()
