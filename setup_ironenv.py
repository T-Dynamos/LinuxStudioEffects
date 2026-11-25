#!/usr/bin/env python3
"""Self-contained IRON / MLIR-AIE setup and NPU verification script.

Provisions an isolated Python virtual environment (``ironenv``) using the
standalone HRX runtime (``libhrx.so``) and bundled ``hrx-xclbinutil`` so NPU
code runs directly against ``/dev/accel/accel*`` without system XRT or
``xrt-plugin-amdxdna`` packages.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import json
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

SUPPORTED_PY_MINORS = (14, 13, 12, 11)
MLIR_AIE_REPO = "https://github.com/Xilinx/mlir-aie.git"
MLIR_AIE_LATEST_WHEELS = "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-5"
MLIR_AIE_RELEASE_URL = "https://github.com/Xilinx/mlir-aie/releases"
LLVM_AIE_NIGHTLY = "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
HRX_RELEASES_URL = "https://github.com/jtuyls/hrx/releases"

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
    has_cmake: bool
    cc_bin: str
    cxx_bin: str
    zstd_mode: str


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
    """Locate a host Python 3.11-3.14 executable."""
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


def detect_zstd_mode() -> str | None:
    """Determine how to unpack .tar.zst archives without extra packages."""
    try:
        subprocess.run(
            ["tar", "--zstd", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return "tar_zstd"
    except (OSError, subprocess.SubprocessError):
        pass
    if shutil.which("zstd"):
        return "zstd_pipe"
    try:
        import compression.zstd  # noqa: F401  # Python 3.14+ stdlib

        return "py_zstd"
    except ImportError:
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
        python_spec = "3.13"
        print("  [OK]   Python: no system 3.11-3.14 found; uv will provision 3.13")
    else:
        print("  [FAIL] Python: need Python 3.11-3.14 or 'uv'")
        die("Install Python 3.11-3.14 or 'uv' (https://docs.astral.sh/uv/).")

    if test_only:
        return PreflightResult(
            accel_nodes=rw_nodes,
            vbnv=vbnv,
            fw_version=fw_version,
            npu2=npu2,
            python_spec=python_spec,
            use_uv=use_uv,
            has_cmake=False,
            cc_bin="",
            cxx_bin="",
            zstd_mode="",
        )

    if not shutil.which("git"):
        print("  [FAIL] Tool: 'git' not found")
        die("'git' is required to clone Xilinx/mlir-aie.")
    print(f"  [OK]   Git: {shutil.which('git')}")

    cc_bin = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang") or ""
    cxx_bin = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++") or ""
    has_cmake = shutil.which("cmake") is not None
    if not (cc_bin and cxx_bin):
        print("  [FAIL] Compiler: C/C++ compiler not found (needed to build hrx-xclbinutil)")
        die("Install gcc/g++ or clang/clang++ to compile standalone hrx-xclbinutil.")
    print(f"  [OK]   Compiler: {cxx_bin}" + (" (+ cmake)" if has_cmake else " (direct build, no cmake)"))

    zstd_mode = detect_zstd_mode() or ""
    if not zstd_mode:
        print("  [FAIL] Archive: neither 'tar --zstd', 'zstd', nor Python 3.14 zstd available")
        die("Install 'zstd' (or tar with zstd support) to unpack libhrx.")
    print(f"  [OK]   Zstd unpacker: {zstd_mode}")

    return PreflightResult(
        accel_nodes=rw_nodes,
        vbnv=vbnv,
        fw_version=fw_version,
        npu2=npu2,
        python_spec=python_spec,
        use_uv=use_uv,
        has_cmake=has_cmake,
        cc_bin=cc_bin,
        cxx_bin=cxx_bin,
        zstd_mode=zstd_mode,
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


def parse_env_file(path: pathlib.Path) -> dict[str, str]:
    """Parse simple KEY="VALUE" lines from utils/hrx-release.env."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def resolve_hrx_asset(mlir_aie_dir: pathlib.Path, *, channel: str) -> tuple[str, str, str | None]:
    """Return (asset_name, asset_url, expected_sha256_or_none)."""
    if channel == "latest":
        try:
            html = fetch_bytes(HRX_RELEASES_URL).decode("utf-8", errors="replace")
            m = re.search(r"/jtuyls/hrx/releases/tag/(flm-hrx-amdxdna-v[^\"<> /]+)", html)
            if m:
                tag = m.group(1)
                assets_html = fetch_bytes(f"{HRX_RELEASES_URL}/expanded_assets/{tag}").decode(
                    "utf-8", errors="replace"
                )
                am = re.search(r"(hrx-[^\"<> /]+linux-x86_64\.tar\.zst)", assets_html)
                if am:
                    asset_name = am.group(1)
                    asset_url = f"{HRX_RELEASES_URL}/download/{tag}/{asset_name}"
                    expected_sha = None
                    if f"{asset_name}.sha256" in assets_html:
                        sha_txt = fetch_bytes(f"{asset_url}.sha256").decode("utf-8", errors="replace")
                        expected_sha = sha_txt.strip().split()[0]
                    return asset_name, asset_url, expected_sha
        except (urllib.error.URLError, OSError, IndexError) as exc:
            warn(f"Could not query latest HRX release ({exc}); falling back to mlir-aie pinned release.")

    pinned = parse_env_file(mlir_aie_dir / "utils" / "hrx-release.env")
    repo = pinned["HRX_RELEASE_REPO"]
    tag = pinned["HRX_RELEASE_TAG"]
    asset_name = pinned["HRX_RELEASE_ASSET"]
    expected_sha = pinned.get("HRX_RELEASE_SHA256")
    return asset_name, f"https://github.com/{repo}/releases/download/{tag}/{asset_name}", expected_sha


def unpack_tar_zst(archive: pathlib.Path, dest_dir: pathlib.Path, zstd_mode: str) -> None:
    """Unpack a .tar.zst archive into dest_dir stripping the top-level directory."""
    if zstd_mode == "tar_zstd":
        run(["tar", "--zstd", "-xf", str(archive), "-C", str(dest_dir), "--strip-components=1"])
        return

    if zstd_mode == "zstd_pipe":
        p1 = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
        try:
            subprocess.run(
                ["tar", "-xf", "-", "-C", str(dest_dir), "--strip-components=1"],
                stdin=p1.stdout,
                check=True,
            )
        finally:
            if p1.stdout:
                p1.stdout.close()
            if p1.wait() != 0:
                die("zstd decompression failed")
        return

    import io
    from compression import zstd

    raw = zstd.decompress(archive.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        members = []
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            m.name = parts[1]
            members.append(m)
        tf.extractall(dest_dir, members=members)


def install_hrx_runtime(
    venv_dir: pathlib.Path,
    mlir_aie_dir: pathlib.Path,
    preflight: PreflightResult,
    *,
    channel: str,
) -> pathlib.Path:
    """Download, verify, and unpack libhrx directly inside venv_dir/hrx."""
    asset_name, asset_url, expected_sha = resolve_hrx_asset(mlir_aie_dir, channel=channel)
    hrx_dir = venv_dir / "hrx"

    with tempfile.TemporaryDirectory() as tmp:
        archive = pathlib.Path(tmp) / asset_name
        log(f"Downloading HRX runtime: {asset_name}")
        fetch_to_file(asset_url, archive)

        if expected_sha:
            actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                die(f"HRX sha256 mismatch: got {actual_sha}, expected {expected_sha}")
            log(f"HRX checksum verified ({actual_sha[:16]}...)")

        if hrx_dir.exists():
            shutil.rmtree(hrx_dir)
        hrx_dir.mkdir(parents=True, exist_ok=True)
        unpack_tar_zst(archive, hrx_dir, preflight.zstd_mode)

    if not (hrx_dir / "lib" / "libhrx.so").exists():
        die(f"libhrx.so missing after extraction in {hrx_dir / 'lib'}")
    log(f"Installed libhrx into {hrx_dir}")
    return hrx_dir


def locate_venv_package_dir(venv_dir: pathlib.Path, dist_name: str, subdir: str) -> pathlib.Path:
    """Resolve the installed package directory inside venv_dir without importing it."""
    code = (
        "import importlib.metadata as md, pathlib, sys; "
        "print(pathlib.Path(md.distribution(sys.argv[1]).locate_file(sys.argv[2])))"
    )
    out = subprocess.check_output(
        [str(venv_dir / "bin" / "python"), "-c", code, dist_name, subdir],
        text=True,
    ).strip()
    path = pathlib.Path(out)
    if not path.exists():
        die(f"Expected package directory not found: {path}")
    return path


def install_xclbinutil(
    venv_dir: pathlib.Path,
    mlir_aie_dir: pathlib.Path,
    preflight: PreflightResult,
) -> pathlib.Path:
    """Build Boost-free, XRT-free hrx-xclbinutil and install it inside ironenv."""
    src_dir = mlir_aie_dir / "third_party" / "hrx-xclbinutil"
    if not src_dir.exists():
        die(f"Missing hrx-xclbinutil sources at {src_dir}")

    mlir_aie_pkg_dir = locate_venv_package_dir(venv_dir, "mlir_aie", "mlir_aie")

    with tempfile.TemporaryDirectory() as tmp:
        build_dir = pathlib.Path(tmp)
        if preflight.has_cmake:
            log("Building standalone hrx-xclbinutil with cmake...")
            run(["cmake", "-S", str(src_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], capture=True)
            jobs = str(os.cpu_count() or 4)
            run(["cmake", "--build", str(build_dir), f"-j{jobs}"], capture=True)
            built_bin = build_dir / "tools" / "hrx-xclbinutil"
        else:
            log("Building standalone hrx-xclbinutil directly with C/C++ compiler...")
            gen_dir = build_dir / "gen"
            obj_dir = build_dir / "obj"
            gen_dir.mkdir(parents=True, exist_ok=True)
            obj_dir.mkdir(parents=True, exist_ok=True)

            ver_tpl = (src_dir / "vendor" / "version.h.in").read_text(encoding="utf-8")
            replacements = {
                "@XRT_VERSION_STRING@": "2.18.0",
                "@XRT_HEAD_COMMITS@": "0xDEADBEEF",
                "@XRT_BRANCH_COMMITS@": "0xDEADFACE",
                "@XRT_BRANCH@": "mlir-aie-hrx",
                "@XRT_HASH@": "",
                "@XRT_HASH_DATE@": "",
                "@XRT_DATE_RFC@": "",
                "@XRT_DATE@": "",
                "@XRT_MODIFIED_FILES@": "",
            }
            for k, v in replacements.items():
                ver_tpl = ver_tpl.replace(k, v)
            (gen_dir / "version.h").write_text(ver_tpl, encoding="utf-8")

            pdi_root = src_dir / "vendor" / "xclbinutil" / "aie-pdi-transform"
            pdi_objs: list[str] = []
            for c_file in sorted([*(pdi_root / "src").glob("*.c"), *(pdi_root / "lib").glob("*.c")]):
                obj = str(obj_dir / f"{c_file.stem}.o")
                run(
                    [
                        preflight.cc_bin,
                        "-O2",
                        "-c",
                        f"-I{pdi_root / 'include'}",
                        f"-I{pdi_root / 'lib'}",
                        str(c_file),
                        "-o",
                        obj,
                    ]
                )
                pdi_objs.append(obj)

            cxx_srcs = sorted(str(p) for p in (src_dir / "vendor" / "xclbinutil").glob("*.cxx"))
            built_bin = build_dir / "xclbinutil"
            run(
                [
                    preflight.cxx_bin,
                    "-O2",
                    "-std=c++17",
                    "-fexceptions",
                    "-frtti",
                    "-w",
                    "-pthread",
                    f"-I{gen_dir}",
                    f"-I{src_dir / 'util'}",
                    f"-I{src_dir / 'vendor' / 'include'}",
                    f"-I{src_dir / 'vendor' / 'xclbinutil'}",
                    f"-I{pdi_root / 'include'}",
                    *cxx_srcs,
                    str(src_dir / "util" / "hrx_util.cpp"),
                    *pdi_objs,
                    "-o",
                    str(built_bin),
                ]
            )

        for target in (venv_dir / "bin" / "xclbinutil", mlir_aie_pkg_dir / "bin" / "xclbinutil"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_bin, target)
            target.chmod(0o755)

    log(f"Installed standalone xclbinutil to {venv_dir / 'bin' / 'xclbinutil'}")
    return venv_dir / "bin" / "xclbinutil"


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
    """Configure bash/zsh, fish, and Python .pth runtime wiring inside ironenv."""
    mlir_aie_pkg_dir = locate_venv_package_dir(venv_dir, "mlir_aie", "mlir_aie")
    peano_pkg_dir = locate_venv_package_dir(venv_dir, "llvm-aie", "llvm-aie")
    purelib = pathlib.Path(
        subprocess.check_output(
            [str(venv_dir / "bin" / "python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            text=True,
        ).strip()
    )

    iron_env_sh = venv_dir / "iron_env.sh"
    iron_env_sh.write_text(
        f"""# Auto-generated by setup_ironenv.py
_ironenv_root="{venv_dir}"
export MLIR_AIE_INSTALL_DIR="{mlir_aie_pkg_dir}"
export PEANO_INSTALL_DIR="{peano_pkg_dir}"
export HRX_DIR="${{_ironenv_root}}/hrx"
export LIBHRX_DIR="${{HRX_DIR}}/lib"
export HRX_LIBHRX="${{LIBHRX_DIR}}/libhrx.so"
export NPU_RUNTIME="hrx"
export NPU2="{preflight.npu2}"
export AIE_XCLBINUTIL="${{_ironenv_root}}/bin/xclbinutil"
export PATH="${{MLIR_AIE_INSTALL_DIR}}/bin:${{_ironenv_root}}/bin:${{PATH}}"
export PYTHONPATH="${{MLIR_AIE_INSTALL_DIR}}/python${{PYTHONPATH:+:${{PYTHONPATH}}}}"
export LD_LIBRARY_PATH="${{LIBHRX_DIR}}:${{MLIR_AIE_INSTALL_DIR}}/lib${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"
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
set -gx HRX_DIR "$_ironenv_root/hrx"
set -gx LIBHRX_DIR "$HRX_DIR/lib"
set -gx HRX_LIBHRX "$LIBHRX_DIR/libhrx.so"
set -gx NPU_RUNTIME "hrx"
set -gx NPU2 "{preflight.npu2}"
set -gx AIE_XCLBINUTIL "$_ironenv_root/bin/xclbinutil"
set -gx PATH "$MLIR_AIE_INSTALL_DIR/bin" "$_ironenv_root/bin" $PATH
if set -q PYTHONPATH
    set -gx PYTHONPATH "$MLIR_AIE_INSTALL_DIR/python" $PYTHONPATH
else
    set -gx PYTHONPATH "$MLIR_AIE_INSTALL_DIR/python"
end
if set -q LD_LIBRARY_PATH
    set -gx LD_LIBRARY_PATH "$LIBHRX_DIR" "$MLIR_AIE_INSTALL_DIR/lib" $LD_LIBRARY_PATH
else
    set -gx LD_LIBRARY_PATH "$LIBHRX_DIR" "$MLIR_AIE_INSTALL_DIR/lib"
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

    # Python startup hook (.pth) so any shell (zsh/nu/csh) or direct venv python invocation works.
    runtime_hook = purelib / "_ironenv_runtime.py"
    runtime_hook.write_text(
        f"""# Auto-generated by setup_ironenv.py
import os
import sys

_VENV_ROOT = {str(venv_dir)!r}
_MLIR_AIE = {str(mlir_aie_pkg_dir)!r}
_PEANO = {str(peano_pkg_dir)!r}
_HRX_DIR = os.path.join(_VENV_ROOT, "hrx")
_LIBHRX_DIR = os.path.join(_HRX_DIR, "lib")
_MLIR_PY = os.path.join(_MLIR_AIE, "python")

os.environ.setdefault("MLIR_AIE_INSTALL_DIR", _MLIR_AIE)
os.environ.setdefault("PEANO_INSTALL_DIR", _PEANO)
os.environ.setdefault("HRX_DIR", _HRX_DIR)
os.environ.setdefault("LIBHRX_DIR", _LIBHRX_DIR)
os.environ.setdefault("HRX_LIBHRX", os.path.join(_LIBHRX_DIR, "libhrx.so"))
os.environ.setdefault("NPU_RUNTIME", "hrx")
os.environ.setdefault("NPU2", {str(preflight.npu2)!r})
os.environ.setdefault("AIE_XCLBINUTIL", os.path.join(_VENV_ROOT, "bin", "xclbinutil"))

_bin_dirs = [os.path.join(_MLIR_AIE, "bin"), os.path.join(_VENV_ROOT, "bin")]
_cur_path = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
os.environ["PATH"] = os.pathsep.join([d for d in _bin_dirs if d not in _cur_path] + _cur_path)

if _MLIR_PY not in sys.path:
    sys.path.insert(0, _MLIR_PY)
""",
        encoding="utf-8",
    )
    (purelib / "ironenv_npu.pth").write_text("import _ironenv_runtime\n", encoding="utf-8")
    log(f"Wired bash, fish, and Python .pth runtime hooks inside {venv_dir}")


def run_npu_tests(venv_dir: pathlib.Path, mlir_aie_dir: pathlib.Path) -> None:
    """Verify JIT compilation and NPU hardware execution inside a clean env."""
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
                    "assert type(t).__name__ == 'HRXTensor', type(t); "
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
echo "[test] NPU_RUNTIME=${{NPU_RUNTIME}} NPU2=${{NPU2}} HRX_LIBHRX=${{HRX_LIBHRX}}"

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
                f'assert type(t).__name__ == \'HRXTensor\'; '
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
        help="Install rolling latest wheels/HRX ('latest') or latest stable release ('release')",
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
        install_hrx_runtime(venv_dir, mlir_aie_dir, preflight, channel=args.channel)
        install_xclbinutil(venv_dir, mlir_aie_dir, preflight)
        write_env_integration(venv_dir, preflight)

    if not args.skip_test:
        run_npu_tests(venv_dir, mlir_aie_dir)

    print(f'\nSetup complete! Activate in any shell with:\n  source "{venv_dir / "bin" / "activate"}"')


if __name__ == "__main__":
    main()
