#!/usr/bin/env python3
"""Self-contained IRON / MLIR-AIE setup with a bundled standalone XRT runtime."""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass

SUPPORTED_PY_MINORS = (14,)
MLIR_AIE_REPO = "https://github.com/Xilinx/mlir-aie.git"
MLIR_AIE_LATEST_WHEELS = "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-5"
MLIR_AIE_RELEASE_URL = "https://github.com/Xilinx/mlir-aie/releases"
LLVM_AIE_NIGHTLY = "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
XRT_BUNDLE_DEFAULT = pathlib.Path("/home/tdynamos/amdnpu/dist/ryzen-ai-xrt-npu1-linux-x86_64.tar")
XRT_BUNDLE_NAME = "ryzen-ai-xrt-npu1-linux-x86_64"

ACTIVATE_MARKER_START = "# >>> ironenv-npu >>>"
ACTIVATE_MARKER_END = "# <<< ironenv-npu <<<"


@dataclass
class PreflightResult:
    """Validated host capabilities discovered before modifying disk."""

    accel_nodes: list[str]
    vbnv: str
    fw_version: str
    npu2: int
    python_spec: str
    use_uv: bool


def log(msg: str) -> None:
    print(f"[ironenv] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[ironenv] WARNING: {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> None:
    print(f"[ironenv] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def run(
    cmd: list[str],
    *,
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess and raise cleanly on failure."""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def fetch_bytes(url: str, timeout: int = 20) -> bytes:
    """Download a URL with a standard User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": "ironenv-setup/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_to_file(url: str, dest: pathlib.Path, timeout: int = 45) -> None:
    """Stream a URL to a local file, preferring curl with retries if available."""
    if shutil.which("curl"):
        run(
            [
                "curl",
                "-fL",
                "--connect-timeout",
                "15",
                "--max-time",
                str(timeout),
                "--retry",
                "2",
                "-o",
                str(dest),
                url,
            ]
        )
        return
    req = urllib.request.Request(url, headers={"User-Agent": "ironenv-setup/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def read_sysfs_attr(device_dir: pathlib.Path, name: str) -> str:
    """Return stripped sysfs attribute content or empty string if missing."""
    path = device_dir / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def find_compatible_python() -> str | None:
    """Locate CPython 3.14, which matches the XRT bundle's pyxrt binding."""
    cur_major, cur_minor = sys.version_info[:2]
    if cur_major == 3 and cur_minor in SUPPORTED_PY_MINORS:
        return sys.executable
    for minor in SUPPORTED_PY_MINORS:
        for name in (f"python3.{minor}", "python3"):
            found = shutil.which(name)
            if not found:
                continue
            try:
                out = subprocess.check_output(
                    [found, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                    text=True,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                continue
            if out == f"3.{minor}":
                return found
    return None


def preflight_checks(*, test_only: bool) -> PreflightResult:
    """Detect NPU device nodes, kernel driver, permissions, and build tools before setup."""
    print("[preflight] Checking host OS, NPU device, permissions, and toolchain...")

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        die(f"Unsupported platform: {platform.system()} {platform.machine()} (requires Linux x86_64)")
    print(f"  [OK]   Platform: {platform.system()} {platform.machine()} (kernel {platform.release()})")

    nodes = sorted(glob.glob("/dev/accel/accel*"))
    if not nodes:
        print("  [FAIL] Device node: no /dev/accel/accel* found")
        die(
            "No NPU accel device found.\n"
            "  Check that:\n"
            "  1. Linux kernel >= 6.14 is running (with in-tree amdxdna driver).\n"
            "  2. The NPU/IPU is enabled in BIOS.\n"
            "  3. 'amdxdna' kernel module is loaded (lsmod | grep amdxdna)."
        )

    rw_nodes = [n for n in nodes if os.access(n, os.R_OK | os.W_OK)]
    if not rw_nodes:
        st = os.stat(nodes[0])
        mode = stat.S_IMODE(st.st_mode)
        print(f"  [FAIL] Permissions: {nodes[0]} (mode {oct(mode)}) is not read/write for user {os.environ.get('USER', '')}")
        die(
            "Current user lacks read/write permission on /dev/accel/accel*.\n"
            "Run this one-time sudo setup, then log out and back in:\n"
            "  sudo groupadd -f render\n"
            "  sudo usermod -aG render \"$USER\"\n"
            "  echo 'SUBSYSTEM==\"accel\", KERNEL==\"accel*\", GROUP=\"render\", MODE=\"0660\"' "
            "| sudo tee /etc/udev/rules.d/99-amdxdna.rules\n"
            "  sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=accel --action=add"
        )
    print(f"  [OK]   Device node: {', '.join(rw_nodes)} (rw accessible)")

    driver_bound = pathlib.Path("/sys/module/amdxdna").exists()
    vbnv = ""
    fw_version = ""
    for sys_dev in sorted(glob.glob("/sys/class/accel/accel*/device")):
        p = pathlib.Path(sys_dev)
        drv = p / "driver"
        if drv.is_symlink() and drv.resolve().name == "amdxdna":
            driver_bound = True
        vbnv = vbnv or read_sysfs_attr(p, "vbnv")
        fw_version = fw_version or read_sysfs_attr(p, "fw_version")

    if not driver_bound:
        print("  [FAIL] Kernel driver: amdxdna module not bound to accel device")
        die("amdxdna kernel driver is not loaded/bound. Try 'sudo modprobe amdxdna'.")
    print(
        f"  [OK]   Kernel driver: amdxdna bound"
        + (f", model={vbnv}" if vbnv else "")
        + (f", firmware={fw_version}" if fw_version else "")
    )

    npu2 = 1 if re.search(r"RyzenAI-npu[456]", vbnv) else 0

    use_uv = shutil.which("uv") is not None
    py_bin = find_compatible_python()
    if py_bin:
        python_spec = py_bin
        py_ver = subprocess.check_output([py_bin, "-c", "import sys; print(sys.version.split()[0])"], text=True).strip()
        print(f"  [OK]   Python: {py_ver} at {py_bin}" + (" (uv available)" if use_uv else ""))
    elif use_uv:
        python_spec = "3.14"
        print("  [OK]   Python: no system Python 3.14 found; uv will provision 3.14")
    else:
        print("  [FAIL] Python: need Python 3.14 or 'uv'")
        die("Install Python 3.14 or 'uv' (https://docs.astral.sh/uv/).")

    if test_only:
        return PreflightResult(
            accel_nodes=rw_nodes,
            vbnv=vbnv,
            fw_version=fw_version,
            npu2=npu2,
            python_spec=python_spec,
            use_uv=use_uv,
        )

    if not shutil.which("git"):
        print("  [FAIL] Tool: 'git' not found")
        die("'git' is required to clone Xilinx/mlir-aie.")
    print(f"  [OK]   Git: {shutil.which('git')}")

    return PreflightResult(
        accel_nodes=rw_nodes,
        vbnv=vbnv,
        fw_version=fw_version,
        npu2=npu2,
        python_spec=python_spec,
        use_uv=use_uv,
    )


def create_or_reuse_venv(venv_dir: pathlib.Path, preflight: PreflightResult, *, recreate: bool) -> None:
    """Create an isolated virtualenv inside venv_dir without system site-packages."""
    if recreate and venv_dir.exists():
        log(f"Removing existing {venv_dir} (--recreate)")
        shutil.rmtree(venv_dir)

    venv_python = venv_dir / "bin" / "python"
    if venv_python.exists():
        log(f"Reusing existing virtualenv at {venv_dir}")
        return

    log(f"Creating isolated virtualenv at {venv_dir}")
    if preflight.use_uv:
        run(["uv", "venv", "--seed", "--python", preflight.python_spec, str(venv_dir)])
    else:
        run([preflight.python_spec, "-m", "venv", str(venv_dir)])


def pip_install(venv_dir: pathlib.Path, preflight: PreflightResult, args: list[str]) -> None:
    """Install Python packages into venv_dir using uv if available, else pip."""
    venv_py = str(venv_dir / "bin" / "python")
    if preflight.use_uv:
        run(["uv", "pip", "install", "--python", venv_py, *args])
    else:
        run([venv_py, "-m", "pip", "install", *args])


def resolve_latest_mlir_aie_release_tag() -> str:
    """Resolve the latest GitHub release tag for Xilinx/mlir-aie without hitting API rate limits."""
    req = urllib.request.Request(
        f"{MLIR_AIE_RELEASE_URL}/latest",
        headers={"User-Agent": "ironenv-setup/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        final_url = resp.geturl()
    tag = final_url.rstrip("/").rsplit("/", 1)[-1]
    if not tag or tag == "releases":
        die(f"Could not resolve latest mlir-aie release tag from {final_url}")
    return tag


def sync_mlir_aie_repo(mlir_aie_dir: pathlib.Path, *, channel: str) -> str:
    """Shallow-clone or fast-forward Xilinx/mlir-aie to main or latest release tag."""
    target_ref = "main" if channel == "latest" else resolve_latest_mlir_aie_release_tag()
    if not (mlir_aie_dir / ".git").exists():
        log(f"Cloning Xilinx/mlir-aie ({target_ref}) into {mlir_aie_dir}")
        run(["git", "clone", "--depth", "1", "--branch", target_ref, MLIR_AIE_REPO, str(mlir_aie_dir)])
    else:
        log(f"Updating Xilinx/mlir-aie checkout to {target_ref}")
        run(["git", "-C", str(mlir_aie_dir), "fetch", "--depth", "1", "origin", target_ref])
        run(["git", "-C", str(mlir_aie_dir), "checkout", "-B", target_ref, "FETCH_HEAD"])
    return target_ref


def install_python_toolchain(
    venv_dir: pathlib.Path,
    mlir_aie_dir: pathlib.Path,
    preflight: PreflightResult,
    *,
    channel: str,
    target_ref: str,
) -> None:
    """Install latest numpy/ml_dtypes/cloudpickle, mlir_aie wheel, and llvm-aie (Peano)."""
    log("Installing Python runtime packages, mlir_aie, and llvm-aie...")
    pip_install(venv_dir, preflight, ["-U", "pip", "numpy>=2.5.3,<3.0", "ml_dtypes>=0.6.0", "cloudpickle"])

    if channel == "latest":
        pip_install(venv_dir, preflight, ["-U", "--pre", "mlir_aie", "-f", MLIR_AIE_LATEST_WHEELS])
        pip_install(venv_dir, preflight, ["-U", "--pre", "llvm-aie", "-f", LLVM_AIE_NIGHTLY])
    else:
        ver = target_ref.lstrip("v")
        pip_install(
            venv_dir,
            preflight,
            ["-U", f"mlir_aie=={ver}", "-f", f"{MLIR_AIE_RELEASE_URL}/expanded_assets/{target_ref}"],
        )
        pip_install(venv_dir, preflight, ["-U", "-r", str(mlir_aie_dir / "utils" / "peano-requirements.txt")])


def sha256_file(path: pathlib.Path) -> str:
    """Hash a large runtime archive without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256(path: pathlib.Path) -> str | None:
    """Read the first SHA-256 token from a conventional .sha256 sidecar."""
    sidecar = pathlib.Path(f"{path}.sha256")
    if not sidecar.is_file():
        return None
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    return parts[0].lower() if parts else None


def install_xrt_runtime(venv_dir: pathlib.Path, bundle_source: str | pathlib.Path) -> pathlib.Path:
    """Verify and unpack the standalone XRT + AMD XDNA bundle into the venv."""
    source_text = str(bundle_source)
    is_url = source_text.startswith(("https://", "http://"))
    xrt_dir = venv_dir / "xrt"

    with tempfile.TemporaryDirectory() as temporary:
        temp_dir = pathlib.Path(temporary)
        if is_url:
            archive = temp_dir / f"{XRT_BUNDLE_NAME}.tar"
            log(f"Downloading standalone XRT bundle from {source_text}")
            fetch_to_file(source_text, archive, timeout=180)
            try:
                sidecar = fetch_bytes(f"{source_text}.sha256", timeout=30).decode("utf-8", errors="replace")
                expected_sha = sidecar.strip().split()[0].lower()
            except (urllib.error.URLError, OSError, IndexError):
                die("XRT bundle URL must have a matching .sha256 sidecar")
        else:
            archive = pathlib.Path(source_text).expanduser().resolve()
            if not archive.is_file():
                die(
                    f"XRT bundle not found at {archive}. Put the bundle there or pass "
                    "--xrt-bundle PATH_OR_URL."
                )
            expected_sha = read_sha256(archive)

        if expected_sha:
            actual_sha = sha256_file(archive)
            if actual_sha != expected_sha:
                die(f"XRT bundle sha256 mismatch: got {actual_sha}, expected {expected_sha}")
            log(f"XRT bundle checksum verified ({actual_sha[:16]}...)")
        else:
            warn("XRT bundle has no .sha256 sidecar; archive integrity was not verified")

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        try:
            with tarfile.open(archive, mode="r:*") as tar:
                tar.extractall(extract_dir, filter="data")
        except (tarfile.TarError, OSError) as exc:
            die(f"Could not extract XRT bundle: {exc}")
        source_xrt = extract_dir / XRT_BUNDLE_NAME / "xrt"
        required = (
            source_xrt / "setup.sh",
            source_xrt / "bin" / "xclbinutil",
            source_xrt / "bin" / "xrt-smi",
            source_xrt / "lib" / "libxrt_core.so",
            source_xrt / "lib" / "libxrt_driver_xdna.so.2",
        )
        missing = [str(path.relative_to(source_xrt)) for path in required if not path.exists()]
        if missing:
            die(f"XRT bundle is missing required files: {', '.join(missing)}")
        if not list((source_xrt / "python").glob("pyxrt*.so")):
            die("XRT bundle does not contain its pyxrt Python binding")

        if xrt_dir.exists():
            shutil.rmtree(xrt_dir)
        shutil.copytree(source_xrt, xrt_dir)

    # Do not leave an old runtime available to mask a broken XRT configuration.
    stale_hrx = venv_dir / "hrx"
    if stale_hrx.exists():
        shutil.rmtree(stale_hrx)

    py_version = subprocess.check_output(
        [str(venv_dir / "bin" / "python"), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True,
    ).strip()
    if py_version != "3.14":
        die(f"Bundled pyxrt targets CPython 3.14, but the environment uses Python {py_version}")

    env = os.environ.copy()
    env["XILINX_XRT"] = str(xrt_dir)
    env["XRT_ROOT"] = str(xrt_dir)
    env["LD_LIBRARY_PATH"] = f"{xrt_dir / 'lib'}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    version = subprocess.run(
        [str(xrt_dir / "bin" / "xclbinutil"), "--version"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if version.returncode:
        die(f"Bundled xclbinutil failed to start:\n{version.stdout}")
    log(f"Installed standalone XRT runtime into {xrt_dir}")
    log(f"Bundled xclbinutil: {version.stdout.strip().splitlines()[0]}")

    # Confirm both the binary runtime and MLIR-AIE's selected Python backend
    # are using this bundle before declaring installation successful.
    binding_env = env.copy()
    binding_env["NPU_RUNTIME"] = "xrt"
    binding_env["PYTHONPATH"] = os.pathsep.join(
        [str(xrt_dir / "python"), binding_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    binding = subprocess.run(
        [
            str(venv_dir / "bin" / "python"),
            "-c",
            (
                "import pyxrt; "
                "from aie.utils.tensor_factory import _NPU_RUNTIME, DEFAULT_TENSOR_CLASS; "
                "assert _NPU_RUNTIME == 'xrt', _NPU_RUNTIME; "
                "assert DEFAULT_TENSOR_CLASS.__name__ == 'XRTTensor', DEFAULT_TENSOR_CLASS; "
                "print('pyxrt:', pyxrt.__file__); "
                "print('MLIR-AIE backend:', _NPU_RUNTIME, DEFAULT_TENSOR_CLASS.__name__)"
            ),
        ],
        env=binding_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if binding.returncode:
        die(f"Standalone XRT Python binding/backend check failed:\n{binding.stdout}")
    for line in binding.stdout.splitlines():
        log(line)
    return xrt_dir


def locate_venv_package_dir(venv_dir: pathlib.Path, dist_name: str, subdir: str) -> pathlib.Path:
    """Resolve an installed package directory inside the target venv."""
    code = (
        "import importlib.metadata as md, pathlib, sys; "
        "print(pathlib.Path(md.distribution(sys.argv[1]).locate_file(sys.argv[2])))"
    )
    try:
        output = subprocess.check_output(
            [str(venv_dir / "bin" / "python"), "-c", code, dist_name, subdir],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        die(f"Could not locate installed distribution {dist_name!r}:\n{exc.output}")
    path = pathlib.Path(output)
    if not path.exists():
        die(f"Expected package directory not found for {dist_name!r}: {path}")
    return path


def _upsert_hook(path: pathlib.Path, hook_body: str) -> None:
    """Idempotently insert or replace the ironenv-npu hook block in an activate script."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"\n?{re.escape(ACTIVATE_MARKER_START)}.*?{re.escape(ACTIVATE_MARKER_END)}\n?",
        re.DOTALL,
    )
    text = pattern.sub("\n", text).rstrip()
    block = f"\n{ACTIVATE_MARKER_START}\n{hook_body.rstrip()}\n{ACTIVATE_MARKER_END}\n"
    path.write_text(text + "\n" + block, encoding="utf-8")


def write_env_integration(venv_dir: pathlib.Path, preflight: PreflightResult) -> None:
    """Wire the bundled XRT libraries and pyxrt into shell and direct Python runs."""
    mlir_aie_pkg_dir = locate_venv_package_dir(venv_dir, "mlir_aie", "mlir_aie")
    peano_pkg_dir = locate_venv_package_dir(venv_dir, "llvm-aie", "llvm-aie")
    purelib = pathlib.Path(
        subprocess.check_output(
            [str(venv_dir / "bin" / "python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            text=True,
        ).strip()
    )
    xrt_dir = venv_dir / "xrt"

    iron_env_sh = venv_dir / "iron_env.sh"
    iron_env_sh.write_text(
        f"""# Auto-generated by setup_ironenv.py
_ironenv_root="{venv_dir}"
export MLIR_AIE_INSTALL_DIR="{mlir_aie_pkg_dir}"
export PEANO_INSTALL_DIR="{peano_pkg_dir}"
export XRT_ROOT="${{_ironenv_root}}/xrt"
export XILINX_XRT="${{XRT_ROOT}}"
export NPU_RUNTIME="xrt"
export NPU2="{preflight.npu2}"
export AIE_XCLBINUTIL="${{XRT_ROOT}}/bin/xclbinutil"
export PATH="${{MLIR_AIE_INSTALL_DIR}}/bin:${{_ironenv_root}}/bin:${{XRT_ROOT}}/bin:${{PATH}}"
export PYTHONPATH="${{XRT_ROOT}}/python:${{MLIR_AIE_INSTALL_DIR}}/python${{PYTHONPATH:+:${{PYTHONPATH}}}}"
export LD_LIBRARY_PATH="${{XRT_ROOT}}/lib:${{MLIR_AIE_INSTALL_DIR}}/lib${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"
unset _ironenv_root
""",
        encoding="utf-8",
    )

    iron_env_fish = venv_dir / "iron_env.fish"
    iron_env_fish.write_text(
        f"""# Auto-generated by setup_ironenv.py
set -l _ironenv_root "{venv_dir}"
set -gx MLIR_AIE_INSTALL_DIR "{mlir_aie_pkg_dir}"
set -gx PEANO_INSTALL_DIR "{peano_pkg_dir}"
set -gx XRT_ROOT "$_ironenv_root/xrt"
set -gx XILINX_XRT "$XRT_ROOT"
set -gx NPU_RUNTIME "xrt"
set -gx NPU2 "{preflight.npu2}"
set -gx AIE_XCLBINUTIL "$XRT_ROOT/bin/xclbinutil"
set -gx PATH "$MLIR_AIE_INSTALL_DIR/bin" "$_ironenv_root/bin" "$XRT_ROOT/bin" $PATH
if set -q PYTHONPATH
    set -gx PYTHONPATH "$XRT_ROOT/python" "$MLIR_AIE_INSTALL_DIR/python" $PYTHONPATH
else
    set -gx PYTHONPATH "$XRT_ROOT/python" "$MLIR_AIE_INSTALL_DIR/python"
end
if set -q LD_LIBRARY_PATH
    set -gx LD_LIBRARY_PATH "$XRT_ROOT/lib" "$MLIR_AIE_INSTALL_DIR/lib" $LD_LIBRARY_PATH
else
    set -gx LD_LIBRARY_PATH "$XRT_ROOT/lib" "$MLIR_AIE_INSTALL_DIR/lib"
end
""",
        encoding="utf-8",
    )

    _upsert_hook(
        venv_dir / "bin" / "activate",
        'if [ -f "${VIRTUAL_ENV}/iron_env.sh" ]; then\n    . "${VIRTUAL_ENV}/iron_env.sh"\nfi',
    )
    _upsert_hook(
        venv_dir / "bin" / "activate.fish",
        'if test -f "$VIRTUAL_ENV/iron_env.fish"\n    source "$VIRTUAL_ENV/iron_env.fish"\nend',
    )

    # A .pth startup hook also supports direct venv Python use without activation.
    runtime_hook = purelib / "_ironenv_runtime.py"
    runtime_hook.write_text(
        f"""# Auto-generated by setup_ironenv.py
import os
import sys

_VENV_ROOT = {str(venv_dir)!r}
_MLIR_AIE = {str(mlir_aie_pkg_dir)!r}
_PEANO = {str(peano_pkg_dir)!r}
_XRT_ROOT = os.path.join(_VENV_ROOT, "xrt")
_MLIR_PY = os.path.join(_MLIR_AIE, "python")
_XRT_PY = os.path.join(_XRT_ROOT, "python")

os.environ["MLIR_AIE_INSTALL_DIR"] = _MLIR_AIE
os.environ["PEANO_INSTALL_DIR"] = _PEANO
os.environ["XRT_ROOT"] = _XRT_ROOT
os.environ["XILINX_XRT"] = _XRT_ROOT
os.environ["NPU_RUNTIME"] = "xrt"
os.environ["NPU2"] = {str(preflight.npu2)!r}
os.environ["AIE_XCLBINUTIL"] = os.path.join(_XRT_ROOT, "bin", "xclbinutil")

_bin_dirs = [os.path.join(_MLIR_AIE, "bin"), os.path.join(_VENV_ROOT, "bin"), os.path.join(_XRT_ROOT, "bin")]
_cur_path = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
os.environ["PATH"] = os.pathsep.join([d for d in _bin_dirs if d not in _cur_path] + _cur_path)
_lib_dirs = [os.path.join(_XRT_ROOT, "lib"), os.path.join(_MLIR_AIE, "lib")]
_cur_libs = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if os.environ.get("LD_LIBRARY_PATH") else []
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([d for d in _lib_dirs if d not in _cur_libs] + _cur_libs)
for _python_dir in (_XRT_PY, _MLIR_PY):
    if _python_dir not in sys.path:
        sys.path.insert(0, _python_dir)
""",
        encoding="utf-8",
    )
    (purelib / "ironenv_npu.pth").write_text("import _ironenv_runtime\n", encoding="utf-8")
    log(f"Wired bash, fish, and Python startup hooks for bundled XRT in {venv_dir}")


def run_npu_tests(venv_dir: pathlib.Path, mlir_aie_dir: pathlib.Path) -> None:
    """Verify XRT tensor allocation and NPU kernel execution inside a clean env."""
    log("Running NPU hardware verification tests in isolated environment...")
    with tempfile.TemporaryDirectory() as cache_dir:
        clean_env = {
            "HOME": os.environ.get("HOME", "/tmp"),
            "USER": os.environ.get("USER", ""),
            "TERM": os.environ.get("TERM", "xterm-256color"),
            "PATH": "/usr/bin:/bin",
            "IRON_CACHE_DIR": cache_dir,
        }
        # Test 1: direct venv python without any activate script (via .pth hook)
        run(
            [
                str(venv_dir / "bin" / "python3"),
                "-c",
                (
                    "import aie.iron as iron, numpy as np; "
                    "dev = iron.ensure_current_device(); "
                    "t = iron.arange(1, 17, dtype=np.int32, device='npu'); "
                    "assert type(t).__name__ == 'XRTTensor', type(t); "
                    "print('[test] Direct python (no activate):', dev, type(t).__name__)"
                ),
            ],
            env=clean_env,
        )

        # Test 2: bash activate + full JIT examples
        script = f"""
set -euo pipefail
source "{venv_dir / 'bin' / 'activate'}"

echo "[test] python: $(which python3) ($(python3 -c 'import sys; print(sys.version.split()[0])'))"
echo "[test] xclbinutil: $(which xclbinutil)"
echo "[test] NPU_RUNTIME=${{NPU_RUNTIME}} NPU2=${{NPU2}} XRT_ROOT=${{XRT_ROOT}}"

echo "[test] Running vector_scalar_add (fresh JIT compile + NPU run)..."
python3 "{mlir_aie_dir / 'programming_examples' / 'basic' / 'vector_scalar_add' / 'vector_scalar_add.py'}" --problem-size 2048

echo "[test] Running vector_vector_add (fresh JIT compile + NPU run)..."
python3 "{mlir_aie_dir / 'programming_examples' / 'basic' / 'vector_vector_add' / 'vector_vector_add.py'}"
"""
        run(["bash", "-c", script], env=clean_env)

        # Test 3: fish activate.fish if fish is installed
        if shutil.which("fish"):
            fish_cmd = (
                f'source "{venv_dir / "bin" / "activate.fish"}"; '
                f'python3 -c "import aie.iron as iron, numpy as np; '
                f't = iron.zeros(8, dtype=np.int32, device=\'npu\'); '
                f'assert type(t).__name__ == \'XRTTensor\'; '
                f'print(\'[test] Fish activate.fish:\', iron.ensure_current_device(), type(t).__name__)"'
            )
            run(["fish", "-c", fish_cmd], env=clean_env)
    log("All NPU tests passed!")


def parse_args() -> argparse.Namespace:
    script_dir = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Set up a self-contained IRON / MLIR-AIE environment in iron-env and test it on the NPU."
    )
    parser.add_argument("--venv", type=pathlib.Path, default=script_dir / "iron-env", help="Target venv directory")
    parser.add_argument(
        "--mlir-aie-dir",
        type=pathlib.Path,
        default=script_dir / "mlir-aie",
        help="Xilinx/mlir-aie checkout directory",
    )
    parser.add_argument(
        "--channel",
        choices=("latest", "release"),
        default="latest",
        help="Install rolling latest MLIR-AIE wheels ('latest') or latest stable release ('release')",
    )
    parser.add_argument(
        "--xrt-bundle",
        default=str(XRT_BUNDLE_DEFAULT),
        help="Standalone XRT .tar path or URL (default: bundled tarball path)",
    )
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the venv from scratch")
    parser.add_argument("--check-only", action="store_true", help="Run /dev/accel and host preflight checks only")
    parser.add_argument("--skip-test", action="store_true", help="Skip NPU verification tests")
    parser.add_argument("--test-only", action="store_true", help="Only run NPU verification tests on existing venv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    venv_dir = args.venv.resolve()
    mlir_aie_dir = args.mlir_aie_dir.resolve()

    preflight = preflight_checks(test_only=args.test_only)
    if args.check_only:
        log("Preflight checks passed (--check-only).")
        return

    if not args.test_only:
        create_or_reuse_venv(venv_dir, preflight, recreate=args.recreate)
        target_ref = sync_mlir_aie_repo(mlir_aie_dir, channel=args.channel)
        install_python_toolchain(venv_dir, mlir_aie_dir, preflight, channel=args.channel, target_ref=target_ref)
        install_xrt_runtime(venv_dir, args.xrt_bundle)
        write_env_integration(venv_dir, preflight)

    if not args.skip_test:
        run_npu_tests(venv_dir, mlir_aie_dir)

    print(f'\nSetup complete! Activate in any shell with:\n  source "{venv_dir / "bin" / "activate"}"')


if __name__ == "__main__":
    main()
