import subprocess, sys, time
t = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn", "esm"], capture_output=True, text=True)
print("pip rc", r.returncode, f"{time.time() - t:.0f}s")
print(r.stdout[-1200:])
print(r.stderr[-2500:])
import importlib
for m in ("umap", "esm", "torch", "sklearn", "numpy", "pandas", "scipy", "matplotlib"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "IMPORT FAIL", repr(e)[:200])
