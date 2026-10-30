import os, subprocess, time
log = open('/content/rs/out/progress.log', 'a')
t = time.time()
log.write(time.strftime('%H:%M:%S') + ' START full\n'); log.flush()
p = subprocess.Popen(['python', '-u', 'rs03_run.py', '--data', '/content/rs/data', '--out', '/content/rs/out'],
                     cwd='/content/rs/code', stdout=log, stderr=subprocess.STDOUT)
p.wait()
log.write(time.strftime('%H:%M:%S') + f' FULL_DONE rc={p.returncode} {time.time() - t:.0f}s\n'); log.flush()
print('FULL_DONE', p.returncode)
