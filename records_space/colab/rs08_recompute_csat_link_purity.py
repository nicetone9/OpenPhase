"""rs08（Colab）：csat_link 修正（UniProt 异构体后缀与多编号归一）之后，重算 diagnostics.json 里"csat link"这一行的近邻纯度。
其余诊断不依赖 csat_link（它不是嵌入输入），不需要重算。"""
import json
import numpy as np
import pandas as pd
import rslib

OUT = "/content/rs/out"
df = rslib.load_records("/content/rs/data/rs_records.tsv")
lab = pd.Series(df["csat_link"].astype(str))
d = json.load(open(f"{OUT}/diagnostics.json"))
for lens in ("full", "prot", "cond", "fullM"):
    idx = np.load(f"{OUT}/knn_{lens}.npz")["idx"][:, :15]
    d["purity_k15"]["csat link"][lens] = list(rslib.purity(idx, lab))
d["purity_k15_note"] = "csat link 一行在 csat_link 修正后重算（rs08）"
json.dump(d, open(f"{OUT}/diagnostics.json", "w"), indent=1)
print("RS08_DONE", d["purity_k15"]["csat link"], flush=True)
