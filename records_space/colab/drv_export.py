import os, subprocess, sys, time
os.makedirs('/content/rs/export', exist_ok=True)
log = open('/content/rs/out/progress.log', 'a')
def run(args, name, cwd='/content/rs/code'):
    log.write(time.strftime('%H:%M:%S') + ' START ' + name + '\n'); log.flush()
    p = subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT); p.wait()
    log.write(time.strftime('%H:%M:%S') + f' DONE {name} rc={p.returncode}\n'); log.flush(); return p.returncode
r0 = run([sys.executable, '-m', 'pip', 'install', '-q', 'esm'], 'pip esm')
r1 = run([sys.executable, '-m', 'unittest', 'test_rslib'], 'unittest')
r2 = run([sys.executable, '-u', 'rs09_export_explorer_data.py', '--data', '/content/rs/data', '--out', '/content/rs/out', '--export', '/content/rs/export', '--examples', '/content/rs/data/unseen_examples.fasta'], 'export')
log.write(time.strftime('%H:%M:%S') + f' EXPORT_FINISHED {r0} {r1} {r2}\n'); log.flush()
print('EXPORT_FINISHED', r0, r1, r2)
