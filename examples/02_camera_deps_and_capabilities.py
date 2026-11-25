"""Example 02: Check camera/V4L2 dependencies and summarize all video devices."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from linux_studio_effects.video import (
    check_camera_deps,
    format_all_camera_summaries,
    format_deps_report,
    summarize_all_cameras,
)


def main() -> int:
    report = check_camera_deps()
    print(format_deps_report(report))
    print()
    summaries = summarize_all_cameras(include_metadata=True)
    print("=== Detailed Capabilities Summary ===")
    print(format_all_camera_summaries(summaries))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
