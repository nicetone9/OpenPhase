import os, subprocess, time
os.makedirs('/content/rs/out', exist_ok=True)
log = open('/content/rs/out/progress.log', 'a')
def run(args, name):
    t = time.time()
    log.write(time.strftime('%H:%M:%S') + ' START ' + name + '\n'); log.flush()
    p = subprocess.Popen(['python', '-u'] + args, cwd='/content/rs/code', stdout=log, stderr=subprocess.STDOUT)
    p.wait()
    log.write(time.strftime('%H:%M:%S') + f' DONE {name} rc={p.returncode} {time.time() - t:.0f}s\n'); log.flush()
    return p.returncode
rc = run(['rs02_embed_new_seqs.py', '--data', '/content/rs/data', '--out', '/content/rs/out', '--main-npy', '/content/rs/data/esmc_L18.npy'], 'rs02')
log.write(time.strftime('%H:%M:%S') + f' RS02_DONE {rc}\n'); log.flush()
print('RS02_DONE', rc)
