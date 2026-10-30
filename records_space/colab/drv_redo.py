import os, subprocess, time
log = open('/content/rs/out/progress.log', 'a')
def run(args, name):
    log.write(time.strftime('%H:%M:%S') + ' START ' + name + '\n'); log.flush()
    p = subprocess.Popen(['python', '-u'] + args, cwd='/content/rs/code', stdout=log, stderr=subprocess.STDOUT); p.wait()
    log.write(time.strftime('%H:%M:%S') + f' DONE {name} rc={p.returncode}\n'); log.flush(); return p.returncode
r1 = run(['rs08_recompute_csat_link_purity.py'], 'rs08')
r2 = run(['rs04_figs.py', '--data', '/content/rs/data', '--out', '/content/rs/out', '--figs', '/content/rs/figs'], 'figs4')
log.write(time.strftime('%H:%M:%S') + f' REDO_FINISHED {r1} {r2}\n'); log.flush()
print('REDO_FINISHED', r1, r2)
