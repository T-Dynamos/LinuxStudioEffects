#!/usr/bin/env python3
"""Measure and display live AMD RyzenAI NPU usage via `xrt-smi` + `sysfs` using `rich`."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _resolve_xrt_env(xrt_dir: str | None = None) -> tuple[str, dict[str, str]]:
    """Locate `.ironenv/xrt` (or `$XILINX_XRT` / `PATH`) and build an XRT-safe subprocess env."""
    env = os.environ.copy()
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    candidates: list[pathlib.Path] = []
    if xrt_dir:
        candidates.append(pathlib.Path(xrt_dir))
    if env.get("XILINX_XRT"):
        candidates.append(pathlib.Path(env["XILINX_XRT"]))
    candidates.append(repo_root / ".ironenv" / "xrt")

    for base in candidates:
        smi = base / "bin" / "xrt-smi"
        if smi.is_file():
            env["XILINX_XRT"] = str(base)
            env["PATH"] = f"{base / 'bin'}:{env.get('PATH', '')}"
            ld = f"{base / 'lib'}"
            if env.get("LD_LIBRARY_PATH"):
                ld = f"{ld}:{env['LD_LIBRARY_PATH']}"
            env["LD_LIBRARY_PATH"] = ld
            return str(smi), env

    found = shutil.which("xrt-smi", path=env.get("PATH"))
    if found:
        return found, env
    raise FileNotFoundError("Could not find `xrt-smi` in `.ironenv/xrt`, `$XILINX_XRT`, or `PATH`.")


def _proc_name(pid: str) -> str:
    if not pid or pid == "N/A":
        return "N/A"
    try:
        comm = pathlib.Path(f"/proc/{int(pid)}/comm").read_text(encoding="utf-8").strip()
        cmd = pathlib.Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if cmd:
            for tok in reversed(cmd.split()):
                if not tok.startswith("-"):
                    return f"{comm} ({pathlib.Path(tok).name})"
        return comm
    except OSError:
        return "N/A"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_sysfs_power(device_id: str) -> dict[str, Any]:
    """Read kernel amdxdna PCI runtime PM counters and firmware version from sysfs."""
    base = pathlib.Path("/sys/bus/pci/devices") / device_id
    out: dict[str, Any] = {
        "fw_version": "N/A",
        "runtime_status": "unknown",
        "runtime_active_ms": None,
        "runtime_suspended_ms": None,
    }
    if not base.is_dir():
        return out
    try:
        out["fw_version"] = (base / "fw_version").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    pwr = base / "power"
    for key, fname, parse_int in (
        ("runtime_status", "runtime_status", False),
        ("runtime_active_ms", "runtime_active_time", True),
        ("runtime_suspended_ms", "runtime_suspended_time", True),
    ):
        try:
            raw = (pwr / fname).read_text(encoding="utf-8").strip()
            out[key] = _as_int(raw) if parse_int else raw
        except OSError:
            pass
    return out


def _query_xrt_json(xrt_smi: str, env: dict[str, str], device: str | None = None) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="xrt_smi_", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [xrt_smi, "examine", "-r", "platform", "-r", "aie-partitions", "-f", "JSON", "-o", tmp_path, "--force"]
        if device:
            cmd.extend(["-d", device])
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip() or f"xrt-smi exited with {res.returncode}")
        return json.loads(pathlib.Path(tmp_path).read_text(encoding="utf-8"))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _snapshot_metrics(data: dict[str, Any]) -> list[dict[str, Any]]:
    devices_out: list[dict[str, Any]] = []
    for dev in data.get("devices", []):
        dev_id = dev.get("device_id", "unknown")
        plats = dev.get("platforms") or []
        static_region = (plats[0].get("static_region") or {}) if plats else {}
        plat_status = (plats[0].get("status") or {}) if plats else {}
        plat_name = static_region.get("name", "unknown")
        total_cols = _as_int(static_region.get("total_columns"), 0)
        power_mode = plat_status.get("power_mode", "unknown")

        aie = dev.get("aie_partitions") or {}
        partitions = aie.get("partitions") or []
        allocated_cols: set[int] = set()
        contexts: dict[tuple[str, str], dict[str, Any]] = {}
        part_summaries: list[dict[str, Any]] = []

        for part in partitions:
            p_idx = str(part.get("partition_index", "?"))
            start_col = _as_int(part.get("start_col"), 0)
            num_cols = _as_int(part.get("num_cols"), 0)
            cols = list(range(start_col, start_col + num_cols))
            allocated_cols.update(cols)
            hw_ctxs = part.get("hw_contexts") or []
            part_summaries.append({"partition_index": p_idx, "columns": cols, "num_contexts": len(hw_ctxs)})
            for ctx in hw_ctxs:
                pid = str(ctx.get("pid", "N/A"))
                cid = str(ctx.get("context_id", "?"))
                key = (pid, cid)
                sub = _as_int(ctx.get("command_submissions"), 0)
                comp = _as_int(ctx.get("command_completions"), 0)
                contexts[key] = {
                    "pid": pid,
                    "process": _proc_name(pid) if ctx.get("process_name") in (None, "N/A") else str(ctx.get("process_name")),
                    "context_id": cid,
                    "partition_index": p_idx,
                    "status": ctx.get("status", "N/A"),
                    "instr_bo_mem": ctx.get("instr_bo_mem", "N/A"),
                    "submissions": sub,
                    "completions": comp,
                    "in_flight": max(0, sub - comp),
                    "errors": _as_int(ctx.get("errors"), 0),
                    "migrations": _as_int(ctx.get("migrations"), 0),
                    "suspensions": _as_int(ctx.get("suspensions"), 0),
                    "gops": ctx.get("gops", "N/A"),
                    "fps": ctx.get("fps", "N/A"),
                    "latency": ctx.get("latency", "N/A"),
                }

        devices_out.append(
            {
                "device_id": dev_id,
                "platform": plat_name,
                "power_mode": power_mode,
                "sysfs": _read_sysfs_power(dev_id),
                "total_columns": total_cols,
                "allocated_columns": sorted(allocated_cols),
                "column_allocation_pct": (100.0 * len(allocated_cols) / total_cols) if total_cols > 0 else 0.0,
                "total_memory_usage": aie.get("total_memory_usage", "N/A"),
                "partitions": part_summaries,
                "contexts": contexts,
            }
        )
    return devices_out


def _compute_delta(
    prev_list: list[dict[str, Any]],
    curr_list: list[dict[str, Any]],
    dt: float,
    est_kernel_ms: float,
) -> list[dict[str, Any]]:
    prev_by_id = {d["device_id"]: d for d in prev_list}
    reports: list[dict[str, Any]] = []
    for curr in curr_list:
        dev_id = curr["device_id"]
        prev = prev_by_id.get(dev_id)
        prev_ctxs = prev["contexts"] if prev else {}

        ctx_rows: list[dict[str, Any]] = []
        tot_d_sub = 0
        tot_d_comp = 0
        tot_in_flight = 0
        tot_err = 0
        active_ctxs = 0
        hw_latencies: list[float] = []

        for key, c in sorted(curr["contexts"].items(), key=lambda kv: (_as_int(kv[0][0]), _as_int(kv[0][1]))):
            p = prev_ctxs.get(key)
            d_sub = max(0, c["submissions"] - p["submissions"]) if p is not None else 0
            d_comp = max(0, c["completions"] - p["completions"]) if p is not None else 0
            cmd_rate = d_comp / max(dt, 1e-6)
            if d_comp > 0 or d_sub > 0:
                active_ctxs += 1
                lat = _as_float(c.get("latency"))
                if lat is not None and lat > 0.0:
                    hw_latencies.append(lat)
            tot_d_sub += d_sub
            tot_d_comp += d_comp
            tot_in_flight += c["in_flight"]
            tot_err += c["errors"]
            row = dict(c)
            row["delta_submissions"] = d_sub
            row["delta_completions"] = d_comp
            row["cmds_per_s"] = cmd_rate
            ctx_rows.append(row)

        total_cmds_per_s = tot_d_comp / max(dt, 1e-6)
        total_cols = curr["total_columns"]
        n_alloc = len(curr["allocated_columns"])
        col_fraction = (n_alloc / total_cols) if total_cols > 0 else 0.0

        # Use hardware-reported average latency when available; otherwise use calibrated kernel time.
        effective_ms_per_cmd = (sum(hw_latencies) / len(hw_latencies)) if hw_latencies else est_kernel_ms
        latency_source = "xrt-latency" if hw_latencies else "est-kernel-ms"
        est_partition_duty_pct = (
            min(100.0, (total_cmds_per_s * effective_ms_per_cmd / 1000.0) * 100.0)
            if effective_ms_per_cmd > 0.0 and n_alloc > 0
            else 0.0
        )
        est_array_usage_pct = est_partition_duty_pct * col_fraction

        sys_curr = curr["sysfs"]
        sys_prev = prev["sysfs"] if prev else {}
        pcie_active_pct: float | None = None
        if sys_curr.get("runtime_active_ms") is not None and sys_prev.get("runtime_active_ms") is not None:
            d_active_ms = max(0, int(sys_curr["runtime_active_ms"]) - int(sys_prev["runtime_active_ms"]))
            pcie_active_pct = min(100.0, 100.0 * (d_active_ms / max(dt * 1000.0, 1.0)))

        reports.append(
            {
                "device_id": dev_id,
                "platform": curr["platform"],
                "power_mode": curr["power_mode"],
                "fw_version": sys_curr.get("fw_version", "N/A"),
                "runtime_status": sys_curr.get("runtime_status", "unknown"),
                "pcie_runtime_active_pct": pcie_active_pct,
                "window_s": dt,
                "total_columns": total_cols,
                "allocated_columns": curr["allocated_columns"],
                "column_allocation_pct": curr["column_allocation_pct"],
                "partitions": curr["partitions"],
                "total_delta_submissions": tot_d_sub,
                "total_delta_completions": tot_d_comp,
                "cmds_per_s": total_cmds_per_s,
                "effective_ms_per_cmd": effective_ms_per_cmd,
                "latency_source": latency_source,
                "est_partition_duty_pct": est_partition_duty_pct,
                "est_array_usage_pct": est_array_usage_pct,
                "active_contexts": active_ctxs,
                "total_contexts": len(ctx_rows),
                "in_flight": tot_in_flight,
                "total_errors": tot_err,
                "contexts": ctx_rows,
            }
        )
    return reports


def _meter_bar(pct: float, width: int = 24) -> Text:
    clamped = max(0.0, min(100.0, pct))
    filled = int(round(clamped / 100.0 * width))
    color = "green" if clamped < 50.0 else ("yellow" if clamped < 80.0 else "red")
    bar = Text()
    bar.append("█" * filled, style=f"bold {color}")
    bar.append("░" * (width - filled), style="dim")
    bar.append(f" {clamped:5.1f}%", style=f"bold {color}")
    return bar


def _build_rich_view(reports: list[dict[str, Any]], *, show_idle: bool, sample_no: int) -> Group:
    ts = time.strftime("%H:%M:%S")
    panels: list[Any] = []
    for rep in reports:
        cols_str = str(rep["allocated_columns"]) if rep["allocated_columns"] else "[]"
        pm_state = rep["runtime_status"]
        pm_style = "bold green" if pm_state == "active" else "dim yellow"
        pcie_txt = f"{rep['pcie_runtime_active_pct']:.0f}%" if rep["pcie_runtime_active_pct"] is not None else "N/A"

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold cyan", no_wrap=True)
        summary.add_column()
        summary.add_row(
            "Est. NPU Array Usage",
            Text.assemble(
                _meter_bar(rep["est_array_usage_pct"]),
                (f"  (active-partition duty ~{rep['est_partition_duty_pct']:.1f}% across ", "dim"),
                (f"{len(rep['allocated_columns'])}/{rep['total_columns']} cols", "cyan"),
                (f" @ {rep['effective_ms_per_cmd']:.2f}ms/cmd [{rep['latency_source']}])", "dim"),
            ),
        )
        summary.add_row(
            "Allocated Columns",
            Text.assemble(
                _meter_bar(rep["column_allocation_pct"]),
                (f"  {len(rep['allocated_columns'])}/{rep['total_columns']} cols {cols_str}", "dim"),
            ),
        )
        summary.add_row(
            "Command Throughput",
            Text.assemble(
                (f"{rep['cmds_per_s']:6.1f} cmd/s", "bold magenta"),
                (f"  (+{rep['total_delta_completions']} completed / +{rep['total_delta_submissions']} submitted in {rep['window_s']:.2f}s)", "dim"),
                ("  |  ", "dim"),
                (f"active_ctx={rep['active_contexts']}/{rep['total_contexts']}", "cyan"),
                ("  ", ""),
                (f"in_flight={rep['in_flight']}", "yellow" if rep["in_flight"] else "dim"),
                ("  ", ""),
                (f"err={rep['total_errors']}", "bold red" if rep["total_errors"] else "green"),
            ),
        )
        summary.add_row(
            "Device Power / FW",
            Text.assemble(
                ("PM=", "dim"),
                (pm_state, pm_style),
                (f" (PCIe active={pcie_txt})", "dim"),
                (f"  |  mode={rep['power_mode']}  |  fw={rep['fw_version']}  |  sample #{sample_no} @ {ts}", "dim"),
            ),
        )

        ctx_table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 1))
        ctx_table.add_column("PID", justify="right", style="cyan", no_wrap=True)
        ctx_table.add_column("Ctx", justify="right", no_wrap=True)
        ctx_table.add_column("Status", no_wrap=True)
        ctx_table.add_column("BO", justify="right", style="dim", no_wrap=True)
        ctx_table.add_column("Total", justify="right", no_wrap=True)
        ctx_table.add_column("Delta", justify="right", no_wrap=True)
        ctx_table.add_column("Cmd/s", justify="right", no_wrap=True)
        ctx_table.add_column("Err", justify="right", no_wrap=True)
        ctx_table.add_column("Process", max_width=26, overflow="ellipsis", no_wrap=True)

        rows = [r for r in rep["contexts"] if show_idle or r["delta_completions"] > 0 or r["delta_submissions"] > 0]
        if not rows and rep["contexts"]:
            rows = rep["contexts"]

        for r in rows:
            is_busy = r["delta_completions"] > 0 or r["delta_submissions"] > 0
            row_style = "bold white" if is_busy else "dim"
            delta_style = "bold green" if r["delta_completions"] > 0 else "dim"
            rate_style = "bold magenta" if r["cmds_per_s"] > 0 else "dim"
            err_style = "bold red" if r["errors"] > 0 else "green"
            ctx_table.add_row(
                r["pid"],
                r["context_id"],
                r["status"],
                str(r["instr_bo_mem"]),
                str(r["completions"]),
                Text(f"+{r['delta_completions']}", style=delta_style),
                Text(f"{r['cmds_per_s']:.1f}", style=rate_style),
                Text(str(r["errors"]), style=err_style),
                r["process"],
                style=row_style,
            )

        body = Group(summary, ctx_table) if rows else Group(summary, Text("\n  No active HW contexts on NPU.", style="dim italic"))
        panels.append(
            Panel(
                body,
                title=f"[bold white]{rep['platform']}[/] [cyan]({rep['device_id']})[/]",
                border_style="blue" if rep["cmds_per_s"] > 0 else "bright_black",
                padding=(0, 1),
            )
        )
    return Group(*panels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure AMD NPU usage & throughput via `xrt-smi` with Rich output.")
    parser.add_argument("-d", "--device", help="Optional PCIe BDF (e.g. 0000:c4:00.1)")
    parser.add_argument("--xrt-dir", help="Optional XRT directory override (defaults to .ironenv/xrt)")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Sampling window in seconds (default: 1.0)")
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of measurement intervals (0 = infinite until Ctrl+C)")
    parser.add_argument("-w", "--watch", action="store_true", help="Continuously monitor with Rich Live UI until Ctrl+C")
    parser.add_argument(
        "--est-kernel-ms",
        type=float,
        default=1.5,
        help="Estimated avg kernel ms per NPU submission when hardware latency is N/A (default: 1.5)",
    )
    parser.add_argument("--all-contexts", action="store_true", help="Show idle HW contexts in addition to active ones")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary per sample")
    args = parser.parse_args(argv)

    xrt_smi, env = _resolve_xrt_env(args.xrt_dir)
    count = 0 if args.watch else max(1, int(args.count))
    console = Console()

    t_prev = time.perf_counter()
    prev = _snapshot_metrics(_query_xrt_json(xrt_smi, env, args.device))
    sample_idx = 0
    try:
        if args.json:
            while count == 0 or sample_idx < count:
                time.sleep(max(0.05, float(args.interval)))
                t_curr = time.perf_counter()
                curr = _snapshot_metrics(_query_xrt_json(xrt_smi, env, args.device))
                reports = _compute_delta(prev, curr, max(1e-6, t_curr - t_prev), args.est_kernel_ms)
                print(json.dumps(reports, indent=2 if count == 1 else None))
                prev, t_prev = curr, t_curr
                sample_idx += 1
            return 0

        if count == 1:
            time.sleep(max(0.05, float(args.interval)))
            t_curr = time.perf_counter()
            curr = _snapshot_metrics(_query_xrt_json(xrt_smi, env, args.device))
            reports = _compute_delta(prev, curr, max(1e-6, t_curr - t_prev), args.est_kernel_ms)
            console.print(_build_rich_view(reports, show_idle=args.all_contexts, sample_no=1))
            return 0

        initial_reports = _compute_delta(prev, prev, 1.0, args.est_kernel_ms)
        with Live(
            _build_rich_view(initial_reports, show_idle=args.all_contexts, sample_no=0),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            while count == 0 or sample_idx < count:
                time.sleep(max(0.05, float(args.interval)))
                t_curr = time.perf_counter()
                curr = _snapshot_metrics(_query_xrt_json(xrt_smi, env, args.device))
                reports = _compute_delta(prev, curr, max(1e-6, t_curr - t_prev), args.est_kernel_ms)
                sample_idx += 1
                live.update(_build_rich_view(reports, show_idle=args.all_contexts, sample_no=sample_idx))
                prev, t_prev = curr, t_curr
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
