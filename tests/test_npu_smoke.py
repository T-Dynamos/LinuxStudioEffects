"""Smoke test: verify NPU detection and run an IRON kernel on the NPU."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from linux_studio_effects.npu import probe_npu


@iron.jit
def affine_kernel(inp: In, out: Out, *, size: CompileTime[int] = 1024):
    """Compute out[i] = inp[i] * 2 + 1 on one AIE compute tile."""
    chunk_ty = np.ndarray[(64,), np.dtype[np.int32]]
    full_ty = np.ndarray[(size,), np.dtype[np.int32]]

    of_in = ObjectFifo(chunk_ty, name="in")
    of_out = ObjectFifo(chunk_ty, name="out")

    def core_fn(in_fifo, out_fifo):
        for _ in range_(size // 64):
            a = in_fifo.acquire(1)
            b = out_fifo.acquire(1)
            for i in range_(64):
                b[i] = a[i] * 2 + 1
            in_fifo.release(1)
            out_fifo.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def sequence(a_host, b_host, in_prod, out_cons):
        in_prod.fill(a_host)
        out_cons.drain(b_host, wait=True)

    rt = Runtime(sequence, [full_ty, full_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main() -> int:
    status = probe_npu()
    print(
        f"NPU available={status.available} device={status.device_name} "
        f"backend={status.backend} tensor={status.tensor_type}"
    )
    if not status.available:
        print(f"Reason: {status.reason}", file=sys.stderr)
        return 1

    n = 1024
    x = iron.arange(n, dtype=np.int32, device="npu")
    y = iron.zeros_like(x)
    affine_kernel(x, y, size=n)

    expected = x.numpy() * 2 + 1
    actual = y.numpy()
    assert np.array_equal(actual, expected), f"Mismatch: {actual[:8]} != {expected[:8]}"
    print("PASS: NPU smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
