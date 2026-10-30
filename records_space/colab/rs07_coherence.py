"""rs07（Colab）：条件相近的记录在记录空间里是否也相近。
对温度（只取实测）、pH、NaCl（log10(1+mM)）、蛋白浓度（log10 µM）：每条有观测值的记录，取 15 近邻（四个透镜，来自 knn_<lens>.npz），
只计两条都有观测值的邻居对，给出 |Δ| 的中位数、落在容差内的比例，以及随机配对（同一批记录里打乱邻居）的对照。
输出 out/diagnostics_coherence.json。"""
import json
import numpy as np
import pandas as pd
import rslib

OUT = "/content/rs/out"
df = rslib.load_records("/content/rs/data/rs_records.tsv")
num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
tc = np.where(df["st_temp"] == "obs", num("temp"), np.nan)
ph = np.where(df["st_pH"] == "obs", num("pH"), np.nan)
nacl = np.where(df["st_NaCl"].isin(["obs", "zero"]), np.log10(1 + np.clip(num("NaCl"), 0, None)), np.nan)
conc = np.where(df["st_conc"] == "obs", num("conc"), np.nan)
specs = {"temperature (measured, °C)": (tc, 2.0), "pH": (ph, 0.2), "NaCl (log10(1+mM))": (nacl, 0.1), "protein conc. (log10 µM)": (conc, 0.2)}
rng = np.random.default_rng(0)
out = {}
for lens in ("full", "prot", "cond", "fullM"):
    idx = np.load(f"{OUT}/knn_{lens}.npz")["idx"][:, :15]
    r = {}
    for name, (v, tol) in specs.items():
        ok = ~np.isnan(v)
        i_rep = np.repeat(np.arange(len(v)), idx.shape[1])
        j = idx.reshape(-1)
        m = (j >= 0) & ok[i_rep] & ok[np.clip(j, 0, None)]
        d = np.abs(v[i_rep[m]] - v[j[m]])
        # 随机对照：同一批有观测值的记录里随机配对
        oi = np.where(ok)[0]
        a, b = rng.choice(oi, 400000), rng.choice(oi, 400000)
        dr = np.abs(v[a] - v[b])
        r[name] = dict(n_pairs=int(m.sum()), median_abs_diff_neighbours=float(np.median(d)), median_abs_diff_random=float(np.median(dr)),
                       share_within_tol_neighbours=float((d <= tol).mean()), share_within_tol_random=float((dr <= tol).mean()), tol=tol)
    out[lens] = r
    print(lens, {k: (round(v["share_within_tol_neighbours"], 3), round(v["share_within_tol_random"], 3)) for k, v in r.items()}, flush=True)
json.dump(out, open(f"{OUT}/diagnostics_coherence.json", "w"), indent=1)
print("COHERENCE_DONE", flush=True)
