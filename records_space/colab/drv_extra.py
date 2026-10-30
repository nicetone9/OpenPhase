import os, subprocess, time
log = open('/content/rs/out/progress.log', 'a')
t = time.time()
# 先重跑单元测试（rslib.knn 改了），再跑补充诊断
r = subprocess.run(['python', '-m', 'unittest', 'test_rslib'], cwd='/content/rs/code', capture_output=True, text=True)
log.write(time.strftime('%H:%M:%S') + ' UNITTEST rc=%d %s\n' % (r.returncode, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '')); log.flush()
log.write(time.strftime('%H:%M:%S') + ' START extra\n'); log.flush()
p = subprocess.Popen(['python', '-u', 'rs06_extra_diag.py', '--data', '/content/rs/data', '--out', '/content/rs/out'], cwd='/content/rs/code', stdout=log, stderr=subprocess.STDOUT)
p.wait()
log.write(time.strftime('%H:%M:%S') + f' EXTRA_FINISHED rc={p.returncode} {time.time() - t:.0f}s\n'); log.flush()
print('EXTRA_FINISHED', p.returncode)
