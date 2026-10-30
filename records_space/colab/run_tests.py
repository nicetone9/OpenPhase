import subprocess, sys
r = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_rslib"], cwd="/content/rs/code", capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-6000:]); print("rc", r.returncode)
