import os, subprocess, time
log = open('/content/rs/out/progress.log', 'a')
t = time.time()
log.write(time.strftime('%H:%M:%S') + ' START smoke\n'); log.flush()
os.makedirs('/content/rs/out_smoke', exist_ok=True)
import shutil; shutil.copy('/content/rs/out/esmc_L18_new.npy', '/content/rs/out_smoke/esmc_L18_new.npy')
p = subprocess.Popen(['python', '-u', 'rs03_run.py', '--data', '/content/rs/data', '--out', '/content/rs/out_smoke', '--limit', '1500', '--no-tsne', '--no-robust'],
                     cwd='/content/rs/code', stdout=log, stderr=subprocess.STDOUT)
p.wait()
log.write(time.strftime('%H:%M:%S') + f' SMOKE_DONE rc={p.returncode} {time.time() - t:.0f}s\n'); log.flush()
print('SMOKE_DONE', p.returncode)
