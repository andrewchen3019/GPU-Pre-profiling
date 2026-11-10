"""Build and run all three benchmarks in separate processes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from check_memprof import EXPECTED, check_file

HERE = Path(__file__).resolve().parent


def run(command, log, timeout):
    # An argument list preserves paths with spaces and avoids shell expansion.
    result = subprocess.run(command, capture_output=True, text=True,
                            errors="replace", timeout=timeout)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}); see {log}\n"
                           f"{result.stderr.strip()}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("profiler", "cuda"))
    parser.add_argument("--case", choices=("all", *EXPECTED), default="all")
    parser.add_argument("--profiler-dir", type=Path, default=HERE.parent / "memprofiler")
    parser.add_argument("--cc", default="cc", help="C compiler executable (Linux profiler mode)")
    parser.add_argument("--nvcc", default="nvcc", help="CUDA compiler executable")
    parser.add_argument("--output", type=Path, help="Build and results directory")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds allowed per run")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    output = (args.output or HERE / "build" / args.mode).resolve()
    cases = list(EXPECTED) if args.case == "all" else [args.case]

    try:
        output.mkdir(parents=True, exist_ok=True)
        suffix = ".exe" if os.name == "nt" else ""
        binary = output / ("memprof_events" if args.mode == "profiler" else "cuda_memory")
        binary = binary.with_suffix(suffix)
        if args.mode == "profiler":
            if os.name == "nt":
                raise RuntimeError("Profiler mode uses pthread-based C code. Run it in Linux/WSL.")
            profiler = args.profiler_dir.resolve()
            for name in ("memprof.c", "memprof.h"):
                if not (profiler / name).is_file():
                    raise RuntimeError(f"Missing {profiler / name}; use the profiler branch or --profiler-dir.")
            command = [args.cc, "-std=c11", "-Wall", "-Wextra", "-pthread",
                       "-I", str(profiler), str(HERE / "memprof_events.c"),
                       str(profiler / "memprof.c"), "-o", str(binary)]
        else:
            # Shared cudart permits later LD_PRELOAD testing on Linux.
            command = [args.nvcc, "--cudart", "shared", str(HERE / "cuda_memory.cu"),
                       "-o", str(binary)]

        # Remove selected old results before building, so failures cannot reuse them.
        for case in cases:
            for extension in ("json", "log"):
                (output / f"{case}.{extension}").unlink(missing_ok=True)
        print(f"Building {args.mode} benchmark...", flush=True)
        run(command, output / "build.log", max(180, args.timeout))
        for case in cases:
            result = run([str(binary), case], output / f"{case}.log", args.timeout)
            if args.mode == "profiler":
                report = output / f"{case}.json"
                report.write_text(result.stdout, encoding="utf-8")
                check_file(case, report)
                print(f"PASS {case}: measured peak={EXPECTED[case][0]} bytes (synthetic events)")
            else:
                marker = f"CUDA_CALLS_OK case={case} expected_peak_bytes={EXPECTED[case][0]}"
                if marker not in result.stdout.splitlines():
                    raise RuntimeError(f"{case}: missing completion marker; see {output / (case + '.log')}")
                print(f"CUDA_CALLS_OK {case}: real API calls succeeded; profiler not measured")
        print(f"Results: {output}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
