import os, subprocess, time
log = open('/content/rs/out/progress.log', 'a')
t = time.time()
p = subprocess.Popen(['python', '-u', 'rs04_figs.py', '--data', '/content/rs/data', '--out', '/content/rs/out', '--figs', '/content/rs/figs'], cwd='/content/rs/code', stdout=log, stderr=subprocess.STDOUT); p.wait()
log.write(time.strftime('%H:%M:%S') + f' FIGS3_FINISHED rc={p.returncode} {time.time() - t:.0f}s\n'); log.flush()
print('FIGS3_FINISHED', p.returncode)
