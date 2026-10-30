import os, subprocess, sys, time
os.makedirs('/content/rs13/out', exist_ok=True)
log = open('/content/rs13/out/progress.log', 'a')
def run(args, name, cwd='/content/rs13/code'):
    log.write(time.strftime('%H:%M:%S') + ' START ' + name + '\n'); log.flush()
    p = subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT); p.wait()
    log.write(time.strftime('%H:%M:%S') + f' DONE {name} rc={p.returncode}\n'); log.flush(); return p.returncode
r0 = run([sys.executable, '-m', 'pip', 'install', '-q', 'esm', 'umap-learn'], 'pip esm umap')
lim = open('/content/rs13/data/limit.txt').read().strip() if os.path.exists('/content/rs13/data/limit.txt') else '0'
r1 = run([sys.executable, '-u', 'rs15_pu.py', '--data', '/content/rs13/data', '--out', '/content/rs13/out', '--limit', lim], 'rs15')
log.write(time.strftime('%H:%M:%S') + f' ALL_DONE {r0} {r1}\n'); log.flush()
print('ALL_DONE', r0, r1)
