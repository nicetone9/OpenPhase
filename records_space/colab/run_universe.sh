#!/bin/zsh
# 新开一个 GPU 会话（A100 → L4 → T4 依次试），跑 rs12 蛋白宇宙，把输出拉回本机后停掉会话。用法：run_universe.sh <会话名> [limit]
# colab 的输出里带会话代理令牌的 URL，所以滤掉含 colab.dev 的行。
setopt nullglob
S=$1; LIM=${2:-0}
T="$(cd "$(dirname "$0")/.." && pwd)"
TASK2="$T/../task2_single_component_condition"
L="$T/data/universe/colab_out"
mkdir -p $L
for g in A100 L4 T4; do
  echo "$(date +%T) 申请 $g"
  if timeout 300 colab new -s $S --gpu $g 2>&1 | grep -v colab.dev | tail -2 | grep -qiE "error|400|fail|denied|quota"; then
    timeout 60 colab stop -s $S >/dev/null 2>&1
  else
    echo "$(date +%T) 拿到 $g"; echo $g > $L/gpu_type.txt; break
  fi
done
timeout 120 colab restart-kernel -s $S 2>&1 | grep -v colab.dev | tail -0
cat > /tmp/mk_$S.py <<'PY'
import os
for d in ('/content/rs12/data', '/content/rs12/code', '/content/rs12/out'):
    os.makedirs(d, exist_ok=True)
import subprocess
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip() or 'no gpu')
PY
timeout 120 colab exec -s $S --timeout 60 -f /tmp/mk_$S.py 2>&1 | grep -v colab.dev | tail -2
up() { timeout 240 colab upload -s $S "$1" "$2" 2>&1 | grep -v colab.dev | tail -0; }
up "$T/data/universe/universe_ids.tsv" /content/rs12/data/universe_ids.tsv
up "$T/data/rs_sequences.tsv" /content/rs12/data/rs_sequences.tsv
up "$TASK2/work/esmc_L18.npy" /content/rs12/data/esmc_L18.npy
up "$T/out/rs02/esmc_L18_new.npy" /content/rs12/data/esmc_L18_new.npy
echo $LIM > /tmp/limit_$S.txt; up /tmp/limit_$S.txt /content/rs12/data/limit.txt
up "$T/colab/rs12_universe.py" /content/rs12/code/rs12_universe.py
up "$T/colab/drv_universe.py" /content/rs12/code/drv_universe.py
echo "$(date +%T) 上传完成，开始运行（limit=$LIM）"
(timeout 14400 colab exec -s $S --timeout 14000 -f "$T/colab/drv_universe.py" > $L/_exec.log 2>&1 &)
# 同时把 out 里的分片和日志镜像回本机；看到 ALL_DONE 后收尾
zsh "$T/colab/mirror.sh" $S /content/rs12/out $L ALL_DONE
timeout 120 colab stop -s $S 2>&1 | grep -v colab.dev | tail -1
echo RUN_UNIVERSE_ALL_DONE
