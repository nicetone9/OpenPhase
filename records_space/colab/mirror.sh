#!/bin/zsh
# 每 60 秒把 VM 上 out 目录里本机还没有的文件拉回来，并校验（npz 必须是合法 zip、json 必须能解析，不合法就删掉下一轮重拉）；
# progress.log 与 log_*.txt 每轮刷新。看到标记词（默认 ALL_DONE）后再完整拉一遍并退出。拉取错误写在 <本地目录>/_mirror_errors.log。
# 用法：mirror.sh <会话> <远端目录> <本地目录> [标记词]
S=$1; R=$2; L=$3; MARK=${4:-ALL_DONE}
mkdir -p $L
pull() {
  local f=$1
  colab download -s $S "$R/$f" "$L/$f.part" > /dev/null 2> $L/_dl_err.txt
  if [ -f "$L/$f.part" ]; then
    ok=1
    case "$f" in
      *.npz) python3 -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" "$L/$f.part" || ok=0;;
      *.json) python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$L/$f.part" || ok=0;;
    esac
    if [ $ok = 1 ]; then mv "$L/$f.part" "$L/$f"; else rm -f "$L/$f.part"; echo "$(date +%T) invalid $f" >> $L/_mirror_errors.log; fi
  else
    echo "$(date +%T) fail $f: $(grep -v colab.dev $L/_dl_err.txt | tail -1)" >> $L/_mirror_errors.log
  fi
}
for i in $(seq 1 600); do
  cur=$(colab ls -s $S $R 2>/dev/null | grep -v '/$')
  [ -z "$cur" ] && echo "$(date +%T) empty listing" >> $L/_mirror_errors.log
  for f in ${(f)cur}; do
    case "$f" in progress.log|log_*.txt) pull $f;; *) [ -f "$L/$f" ] || pull $f;; esac
  done
  if grep -q "$MARK" "$L/progress.log" 2>/dev/null; then
    for f in ${(f)cur}; do pull $f; done
    echo "mirror finished $(date +%T)"; exit 0
  fi
  sleep 60
done
