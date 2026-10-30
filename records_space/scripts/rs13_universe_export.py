#!/usr/bin/env python3
"""rs13（本机，纯标准库）：把 Colab 上 rs12 的输出（universe.npz）整理成页面用的 explorer/universe_data.json。
npz 是 zip 里的 .npy，本机没有 numpy，所以这里用标准库自己读（支持 float16/32、整数、布尔和 Unicode 字符串）。
页面里的蛋白宇宙：UMAP 坐标（归一到 [0,1]，同其他透镜的做法）、Zn（int8，×127）、名称、位掩码、最近的数据集序列。"""
import ast
import base64
import json
import os
import struct
import sys
import zipfile
from array import array

ROOT = "/Users/lynn/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Projects/LLPS/condition_llps/records_space_2026-10-29"
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/universe/colab_out/universe.npz"
FMT = {"<f2": "e", "<f4": "f", "<f8": "d", "<i1": "b", "|i1": "b", "<u1": "B", "|u1": "B", "|b1": "?", "<i2": "h", "<u2": "H", "<i4": "i", "<u4": "I", "<i8": "q", "<u8": "Q"}


def read_npy(buf):
    assert buf[:6] == b"\x93NUMPY"
    if buf[6] == 1:
        hl, off = struct.unpack("<H", buf[8:10])[0], 10
    else:
        hl, off = struct.unpack("<I", buf[8:12])[0], 12
    h = ast.literal_eval(buf[off:off + hl].decode("latin1"))
    data, descr, shape = buf[off + hl:], h["descr"], h["shape"]
    n = 1
    for s in shape:
        n *= s
    if descr in FMT:
        f = FMT[descr]
        vals = list(struct.unpack("<" + f * n, data[:n * struct.calcsize(f)]))
    elif descr.startswith("<U"):
        w = int(descr[2:])
        vals = [data[i * 4 * w:(i + 1) * 4 * w].decode("utf-32-le").rstrip("\x00") for i in range(n)]
    else:
        raise ValueError(descr)
    return shape, vals


def load_npz(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for nm in z.namelist():
            out[nm[:-4]] = read_npy(z.read(nm))
    return out


def b64(typecode, vals):
    a = array(typecode, vals)
    if sys.byteorder != "little":
        a.byteswap()
    return base64.b64encode(a.tobytes()).decode()


def main():
    z = load_npz(SRC)
    U = z["acc"][0][0]
    xy_u, xy_ds = z["xy_u"][1], z["xy_ds"][1]
    nds = len(xy_ds) // 2
    xs = xy_u[0::2] + xy_ds[0::2]
    ys = xy_u[1::2] + xy_ds[1::2]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    s = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.04
    nx = lambda x: min(max((x - cx) / s + 0.5, 0.0), 1.0)
    ny = lambda y: min(max(1 - ((y - cy) / s + 0.5), 0.0), 1.0)

    def q16(vals_x, vals_y):
        o = []
        for x, y in zip(vals_x, vals_y):
            o.append(round(nx(x) * 65535)); o.append(round(ny(y) * 65535))
        return o
    zn = z["zn"][1]
    zn_i8 = [max(-127, min(127, round(v * 127))) for v in zn]
    def cut(x, n=56):
        if len(x) <= n:
            return x
        c = x[:n].rsplit(" ", 1)[0].rstrip(",;:-(") if " " in x[:n] else x[:n]
        return c + "…"
    name = [cut(x) for x in z["name"][1]]
    out = {"n": U, "ds_n": nds, "acc": z["acc"][1], "name": name, "gene": z["gene"][1], "len": z["length"][1], "nn": [int(v) for v in z["nn_idx"][1]],
           "nncos": [round(float(v) * 100) for v in z["nn_cos"][1]], "trunc": [1 if v else 0 for v in z["trunc"][1]],
           "obs": [1 if v else 0 for v in z["obs"][1]] if "obs" in z else [0] * U, "org": [x[:40] for x in z["org"][1]] if "org" in z else [""] * U,
           "xy": b64("H", q16(xy_u[0::2], xy_u[1::2])), "xyds": b64("H", q16(xy_ds[0::2], xy_ds[1::2])), "zn": b64("b", zn_i8),
           "lo": b64("I", [int(v) for v in z["mask_lo"][1]]), "hi": b64("I", [int(v) for v in z["mask_hi"][1]])}
    miss = os.path.join(os.path.dirname(SRC), "missing_ids.txt")
    out["missing"] = len(open(miss).read().split()) if os.path.exists(miss) else 0
    json.dump(out, open(f"{ROOT}/explorer/universe_data.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print("universe proteins", U, "| dataset sequences", nds, "| missing ids", out["missing"], "| bytes", os.path.getsize(f"{ROOT}/explorer/universe_data.json"))


if __name__ == "__main__":
    main()
