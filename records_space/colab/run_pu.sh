#!/bin/zsh
# 新开 GPU 会话（A100 → L4 → T4 依次试），跑 rs15 的 PU 页地图，把输出拉回本机后停掉会话。用法：run_pu.sh <会话名> [limit]
# colab 的输出里带会话代理令牌的 URL，所以滤掉含 colab.dev 的行。
setopt nullglob
S=$1; LIM=${2:-0}
T="$(cd "$(dirname "$0")/.." && pwd)"
WORLD="$T/../world_and_cluster_maps_2026-10-27/data"
TASK2="$T/../task2_single_component_condition"
L="$T/data/pu/colab_out"
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
for d in ('/content/rs13/data/uni_shards', '/content/rs13/code', '/content/rs13/out'):
    os.makedirs(d, exist_ok=True)
import subprocess
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip() or 'no gpu')
PY
timeout 120 colab exec -s $S --timeout 60 -f /tmp/mk_$S.py 2>&1 | grep -v colab.dev | tail -2
up() { timeout 300 colab upload -s $S "$1" "$2" 2>&1 | grep -v colab.dev | tail -0; }
up "$T/data/pu/pu_manifest.tsv" /content/rs13/data/pu_manifest.tsv
up "$WORLD/bg_emb_L18.npy" /content/rs13/data/bg_emb_L18.npy
up "$WORLD/emb_data_L18.npy" /content/rs13/data/emb_data_L18.npy
for f in $T/data/universe/colab_out/emb_shard_*.npz; do up "$f" /content/rs13/data/uni_shards/$(basename $f); done
echo $LIM > /tmp/limit_$S.txt; up /tmp/limit_$S.txt /content/rs13/data/limit.txt
up "$T/colab/rs12_universe.py" /content/rs13/code/rs12_universe.py
up "$T/colab/rs15_pu.py" /content/rs13/code/rs15_pu.py
up "$T/colab/drv_pu.py" /content/rs13/code/drv_pu.py
echo "$(date +%T) 上传完成，开始运行（limit=$LIM）"
(timeout 14400 colab exec -s $S --timeout 14000 -f "$T/colab/drv_pu.py" > $L/_exec.log 2>&1 &)
zsh "$T/colab/mirror.sh" $S /content/rs13/out $L ALL_DONE
timeout 120 colab stop -s $S 2>&1 | grep -v colab.dev | tail -1
echo RUN_PU_ALL_DONE
