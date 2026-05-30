"""
setup_environment.py
--------------------
Run this script ONCE from Blender's Text Editor before running the import
script. It validates the Python environment and installs any missing packages
into Blender's bundled Python.

Usage:
  1. Open Blender → Scripting workspace
  2. Open this file
  3. Run Script
  4. Check the System Console for results (Window → Toggle System Console)
  5. If any packages were installed, restart Blender before running the
     import script (newly installed packages require a process restart)
"""

import sys
import subprocess
import os


# =============================================================================
# REQUIRED PACKAGES
# Format: (import_name, pip_name)
# =============================================================================

REQUIRED = [
    ("numpy",   "numpy>=1.26"),
    ("scipy",   "scipy>=1.11"),
    ("shapely", "shapely>=2.0"),
]


def check_and_install():
    python = sys.executable
    print(f"\n{'='*60}")
    print(f"F1 Pipeline — Environment Setup")
    print(f"Python : {python}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"{'='*60}\n")

    # ── Ensure pip is available ───────────────────────────────────────────
    print("Bootstrapping pip...", end=" ", flush=True)
    pip_result = subprocess.run(
        [python, "-m", "ensurepip", "--upgrade"],
        capture_output=True, text=True
    )
    if pip_result.returncode == 0:
        print("OK")
    else:
        print(f"warning ({pip_result.stderr.strip()[:80]})")

    # ── Determine install target ──────────────────────────────────────────
    # Use the user's Blender addons/modules directory — this is always
    # writable without admin rights and is on Blender's sys.path.
    import platform
    if platform.system() == "Windows":
        user_modules = os.path.join(
            os.environ.get("APPDATA", ""),
            "Blender Foundation", "Blender", "5.0",
            "scripts", "addons", "modules"
        )
    elif platform.system() == "Darwin":
        user_modules = os.path.expanduser(
            "~/Library/Application Support/Blender/5.0/scripts/addons/modules"
        )
    else:
        user_modules = os.path.expanduser(
            "~/.config/blender/5.0/scripts/addons/modules"
        )

    os.makedirs(user_modules, exist_ok=True)
    print(f"Install target: {user_modules}\n")

    # ── Check which packages are present ─────────────────────────────────
    print("\nChecking packages...\n")
    missing = []
    for import_name, pip_spec in REQUIRED:
        try:
            mod     = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK  {import_name:<12} {version}")
        except ImportError:
            print(f"  --  {import_name:<12} NOT FOUND")
            missing.append((import_name, pip_spec))

    # ── Install missing packages ──────────────────────────────────────────
    if not missing:
        print(f"\nAll packages already installed. Ready to run the import script.")
        print(f"{'='*60}\n")
        return

    print(f"\nInstalling {len(missing)} missing package(s)...\n")
    failed = []
    for import_name, pip_spec in missing:
        print(f"  pip install {pip_spec}")
        result = subprocess.run(
            [python, "-m", "pip", "install", pip_spec,
             "--target", user_modules],
            capture_output=True, text=True
        )
        print(f"    stdout: {result.stdout.strip()[-200:] if result.stdout.strip() else '(none)'}")
        if result.stderr.strip():
            print(f"    stderr: {result.stderr.strip()[-200:]}")
        if result.returncode == 0:
            print(f"    → OK")
        else:
            print(f"    → FAILED")
            failed.append((import_name, pip_spec))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if not failed:
        print("Installation complete.")
        print("\n*** RESTART BLENDER before running the import script ***")
        print("Newly installed packages require a process restart.")
    else:
        print(f"The following packages failed to install:")
        for name, spec in failed:
            print(f"  {spec}")
        print(f"\nTry running this from a terminal instead:")
        print(f'  "{python}" -m pip install {" ".join(s for _, s in failed)} --target "{user_modules}"')
    print(f"{'='*60}\n")


check_and_install()