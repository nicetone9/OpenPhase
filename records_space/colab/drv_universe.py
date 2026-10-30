import os, subprocess, sys, time
os.makedirs('/content/rs12/out', exist_ok=True)
log = open('/content/rs12/out/progress.log', 'a')
def run(args, name, cwd='/content/rs12/code'):
    log.write(time.strftime('%H:%M:%S') + ' START ' + name + '\n'); log.flush()
    p = subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT); p.wait()
    log.write(time.strftime('%H:%M:%S') + f' DONE {name} rc={p.returncode}\n'); log.flush(); return p.returncode
r0 = run([sys.executable, '-m', 'pip', 'install', '-q', 'esm', 'umap-learn'], 'pip esm umap')
lim = open('/content/rs12/data/limit.txt').read().strip() if os.path.exists('/content/rs12/data/limit.txt') else '0'
r1 = run([sys.executable, '-u', 'rs12_universe.py', '--data', '/content/rs12/data', '--out', '/content/rs12/out', '--limit', lim], 'rs12')
log.write(time.strftime('%H:%M:%S') + f' ALL_DONE {r0} {r1}\n'); log.flush()
print('ALL_DONE', r0, r1)
