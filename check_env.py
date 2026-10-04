"""Phase 1 environment check. Run: python check_env.py"""
import importlib
import platform
import sys

REQUIRED = ["numpy", "pandas", "yaml", "sklearn", "matplotlib", "torch",
            "transformers", "peft", "datasets", "accelerate"]
OPTIONAL = ["sentence_transformers", "pytest"]


def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "ok")
    except Exception:
        return None


def main() -> int:
    print(f"python {sys.version.split()[0]} on {platform.platform()}")
    missing = []
    for n in REQUIRED:
        v = ver(n)
        print(f"  {'OK ' if v else 'MISSING'} {n} {v or ''}")
        if not v:
            missing.append(n)
    for n in OPTIONAL:
        v = ver(n)
        print(f"  {'OK ' if v else 'optional-missing'} {n} {v or ''}")
    from utils.reproducibility import get_device
    print("device:", get_device("auto"))
    if missing:
        print("\nMissing required packages:", ", ".join(missing))
        print("Fix: pip install -r requirements.txt")
        return 1
    print("\nEnvironment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
