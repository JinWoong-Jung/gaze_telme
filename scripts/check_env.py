"""Environment sanity check — run after conda env setup.

Usage:
    python scripts/check_env.py
"""

import importlib
import platform
import shutil
import subprocess
import sys

import torch


def print_result(name: str, ok: bool, detail: str) -> bool:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def check_torch() -> bool:
    details = [f"torch={torch.__version__}"]
    details.append(f"cuda={torch.cuda.is_available()}")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    details.append(f"mps={mps_available}")
    return print_result("torch", True, ", ".join(details))


def check_import(module_name: str, required: bool = True) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return print_result(module_name, True, f"imported (version={version})")
    except Exception as exc:
        if required:
            return print_result(module_name, False, f"{type(exc).__name__}: {exc}")
        return print_result(module_name, True, f"optional/missing ({type(exc).__name__}: {exc})")


def check_ffmpeg() -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return print_result("ffmpeg", False, "binary not found in PATH")
    try:
        proc = subprocess.run(
            [ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
        first_line = proc.stdout.splitlines()[0] if proc.stdout else "version check passed"
        return print_result("ffmpeg", True, first_line)
    except Exception as exc:
        return print_result("ffmpeg", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")

    checks = [
        check_torch(),
        check_ffmpeg(),
        check_import("transformers"),
        check_import("cv2"),
        check_import("decord", required=(platform.system() != "Darwin")),
    ]

    if all(checks):
        print("Environment sanity check passed.")
        return 0

    print("Environment sanity check FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
