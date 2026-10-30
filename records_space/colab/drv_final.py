import os, subprocess, time
log = open('/content/rs/out/progress.log', 'a')
t = time.time()
def run(args, name):
    log.write(time.strftime('%H:%M:%S') + ' START ' + name + '\n'); log.flush()
    p = subprocess.Popen(['python', '-u'] + args, cwd='/content/rs/code', stdout=log, stderr=subprocess.STDOUT); p.wait()
    log.write(time.strftime('%H:%M:%S') + f' DONE {name} rc={p.returncode}\n'); log.flush(); return p.returncode
rc1 = run(['rs07_coherence.py'], 'coherence')
rc2 = run(['rs04_figs.py', '--data', '/content/rs/data', '--out', '/content/rs/out', '--figs', '/content/rs/figs'], 'figs2')
log.write(time.strftime('%H:%M:%S') + f' FINAL_FINISHED {rc1} {rc2} {time.time() - t:.0f}s\n'); log.flush()
print('FINAL_FINISHED', rc1, rc2)
