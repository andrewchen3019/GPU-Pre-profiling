"""Check a memprof format-version-1 JSON report against a known benchmark."""
import argparse
import json
from pathlib import Path
import sys

# Independent expected answers, not computed from the reported measurements.
EXPECTED = {
    "single": (64 * 1024 * 1024, 1),
    "sequential": (128 * 1024 * 1024, 2),
    "interleaved": (224 * 1024 * 1024, 3),
}
ANOMALIES = ("unknown_frees", "invalid_args", "duplicate_allocs",
             "null_allocs", "internal_errors")


def expect(obj, key, expected, location="report"):
    if not isinstance(obj, dict) or key not in obj:
        raise ValueError(f"{location}.{key}: missing required field")
    actual = obj[key]
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{location}.{key}: expected {expected!r}, got {actual!r}")


def validate_report(case, report):
    """Raise ValueError on a missing field, wrong value, or wrong type."""
    peak, allocation_count = EXPECTED[case]
    expect(report, "format_version", 1)
    expect(report, "complete", True)
    expect(report, "peak_total_bytes", peak)
    expect(report, "current_total_bytes", 0)
    expect(report, "live_allocations", 0)
    expect(report, "incomplete_reasons", [])
    expect(report, "incomplete_reasons_dropped", 0)

    anomalies = report.get("anomalies")
    for name in ANOMALIES:
        expect(anomalies, name, 0, "anomalies")
    # Reject new nonzero counters too; do not silently hide new anomaly types.
    for name in anomalies:
        expect(anomalies, name, 0, "anomalies")
    for name in ("current_bytes", "peak_bytes"):
        expect(report.get("host_pinned"), name, 0, "host_pinned")

    count = report.get("num_devices")
    devices = report.get("devices")
    if type(count) is not int or count < 1:
        raise ValueError("num_devices must be a positive integer")
    if not isinstance(devices, list) or len(devices) != count:
        raise ValueError("devices must contain exactly num_devices entries")
    seen = set()
    for device in devices:
        if not isinstance(device, dict):
            raise ValueError("each device entry must be an object")
        index = device.get("device")
        if type(index) is not int or not 0 <= index < count or index in seen:
            raise ValueError("device IDs must be unique integers from 0 to num_devices-1")
        seen.add(index)
        active = index == 0
        values = {
            "current_bytes": 0,
            "peak_bytes": peak if active else 0,
            "allocs": allocation_count if active else 0,
            "frees": allocation_count if active else 0,
            "failed_allocs": 0,
            "largest_failed_bytes": 0,
            "managed_current_bytes": 0,
            "managed_peak_bytes": 0,
        }
        for key, expected in values.items():
            expect(device, key, expected, f"device[{index}]")


def check_file(case, path):
    with Path(path).open(encoding="utf-8") as stream:
        validate_report(case, json.load(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=EXPECTED)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        check_file(args.case, args.report)
    except (OSError, ValueError) as error:
        print(f"FAIL {args.case}: {error}", file=sys.stderr)
        return 1
    print(f"PASS {args.case}: peak={EXPECTED[args.case][0]} bytes; clean final state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
