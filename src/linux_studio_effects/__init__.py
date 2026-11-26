"""LinuxStudioEffects: NPU-accelerated camera and microphone effects for Linux."""

import os

# Default BLAS/OpenMP/C thread pools to 1 thread so live video threads and async workers
# do not blast CPU cores or trigger thread-pool spin-wait barriers.
for _env_name, _env_default in (
    ("OPENBLAS_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("LSE_THREADS", "1"),
):
    os.environ.setdefault(_env_name, _env_default)

__version__ = "0.1.0"
