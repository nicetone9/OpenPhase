import os, subprocess, sys, time
os.makedirs('/content/rs6/out', exist_ok=True); os.makedirs('/content/rs6/figs', exist_ok=True)
log = open('/content/rs6/out/progress.log', 'a')
def run(args, name):
    log.write(time.strftime('%H:%M:%S') + ' START ' + name + '\n'); log.flush()
    p = subprocess.run(args, cwd='/content/rs6/code', stdout=log, stderr=subprocess.STDOUT); log.write(time.strftime('%H:%M:%S') + f' DONE {name} rc={p.returncode}\n'); log.flush(); return p.returncode
if not os.path.exists('/content/rs6/.fonts_ok'):
    run(['apt-get', 'install', '-y', '-qq', 'fonts-liberation'], 'apt fonts'); open('/content/rs6/.fonts_ok', 'w').write('1')
run([sys.executable, '-u', 'rs18_coloring_figure.py', '--data', '/content/rs6/data', '--out', '/content/rs6/out', '--figs', '/content/rs6/figs'], 'rs18')
log.write(time.strftime('%H:%M:%S') + ' DRV_END\n'); log.flush()
