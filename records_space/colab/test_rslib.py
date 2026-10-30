"""rslib 的单元测试（需要 numpy / pandas / scikit-learn；本机没有，会跳过。在 Colab 上 python -m unittest）。"""
import unittest

try:
    import numpy as np
    import pandas as pd
    import rslib
    HAVE = True
except Exception:       # pragma: no cover
    HAVE = False


def make_df(n):
    d = {"seq_row": [0] * n}
    for k in rslib.SHORT:
        d[k] = [1.0] * n
        d[f"st_{k}"] = ["obs"] * n
    d["temp"] = [25.0] * n
    d["na_class"] = ["none"] * n
    for k in ("na_len_log10", "na_conc_log10", "na_ratio_log10", "na_gc", "na_purine", "na_maxrun", "na_g4", "na_mfe_per_nt", "na_paired_frac"):
        d[k] = [""] * n
    for k in rslib.NA_FLAGS:
        d[f"na_name_{k}"] = [""] * n
    for k in rslib.BUF:
        d[f"buf_{k}"] = [0] * n
    for k in rslib.ADD:
        d[f"add_{k}"] = [0] * n
    d["st_buffer"] = ["obs"] * n
    d["st_additive"] = ["none"] * n
    df = pd.DataFrame(d)
    df["buf_Tris"] = 1
    return df


def build(df, nseq=4, seed=0):
    rng = np.random.default_rng(seed)
    esmc = rng.normal(size=(nseq, 16)).astype(np.float32)
    inp = rslib.build_inputs(df, esmc, None)
    rslib.calibrate_P(inp, n_pairs=2000)
    return inp


def lens(df, weights=None):
    inp = build(df)
    return rslib.compute_lens_matrices(inp, chunk=3, weights=weights, verbose=False)


@unittest.skipUnless(HAVE, "需要 numpy/pandas/sklearn（Colab）")
class TestLib(unittest.TestCase):
    def test_identical_records_zero(self):
        df = make_df(3)
        L = lens(df)
        for k, M in L.items():
            self.assertTrue(np.allclose(M, 0.0), k)

    def test_symmetry_and_diag(self):
        df = make_df(6)
        df.loc[1, "temp"] = 30.0
        df.loc[2, "st_temp"] = "miss"; df.loc[2, "temp"] = np.nan
        df.loc[3, "na_class"] = "RNA"; df.loc[3, "na_len_log10"] = 2.0
        df.loc[4, "seq_row"] = 2
        df.loc[5, "buf_HEPES"] = 1
        L = lens(df)
        for k, M in L.items():
            self.assertTrue(np.allclose(M, M.T, atol=1e-6), k)
            self.assertTrue(np.allclose(np.diag(M), 0.0), k)
            self.assertTrue((M >= -1e-6).all() and (M <= 1 + 1e-6).all(), k)

    def test_missing_state_vs_observed_only(self):
        df = make_df(2)
        df.loc[1, "st_temp"] = "miss"
        df.loc[1, "temp"] = np.nan
        L = lens(df)
        w = rslib.BLOCK_W
        want = w["C"] * rslib.LAM / 5 / sum(w.values())
        self.assertAlmostEqual(float(L["full"][0, 1]), want, places=6)
        self.assertAlmostEqual(float(L["fullM"][0, 1]), 0.0, places=6)      # M−：缺失不参与

    def test_both_missing_is_zero(self):
        df = make_df(2)
        for i in (0, 1):
            df.loc[i, "st_temp"] = "miss"; df.loc[i, "temp"] = np.nan
        L = lens(df)
        self.assertAlmostEqual(float(L["full"][0, 1]), 0.0, places=6)

    def test_na_presence_is_one(self):
        df = make_df(2)
        df.loc[1, "na_class"] = "RNA"
        L = lens(df)
        w = rslib.BLOCK_W
        self.assertAlmostEqual(float(L["full"][0, 1]), w["N"] / sum(w.values()), places=6)
        self.assertAlmostEqual(float(L["fullM"][0, 1]), w["N"] / sum(w.values()), places=6)   # 适用性不是缺失，M− 里也一样

    def test_temp_state_penalty(self):
        df = make_df(2)
        df.loc[1, "st_temp"] = "rt"
        L = lens(df)
        w = rslib.BLOCK_W
        want = w["C"] * rslib.TEMP_STATE_PEN / 5 / sum(w.values())
        self.assertAlmostEqual(float(L["full"][0, 1]), want, places=6)

    def test_jaccard(self):
        A = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        j = rslib._jac(A, 0, 4)
        self.assertAlmostEqual(float(j[0, 1]), 1 - 1 / 3, places=6)
        self.assertAlmostEqual(float(j[2, 3]), 0.0, places=6)           # 两个空集合
        self.assertAlmostEqual(float(j[0, 2]), 1.0, places=6)

    def test_scalar_dim_rule(self):
        a = np.array([0.0, 0.2, np.nan, np.nan], dtype=np.float32)
        p, o, b = rslib._scal(a, 0, 4)
        self.assertAlmostEqual(float(p[0, 1]), 0.2, places=6)
        self.assertAlmostEqual(float(p[0, 2]), rslib.LAM, places=6)
        self.assertAlmostEqual(float(p[2, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(o[0, 2]), 0.0, places=6)
        self.assertFalse(bool(b[0, 2]))

    def test_knn_masks(self):
        D = np.array([[0, 1, 2, 3], [1, 0, 1, 4], [2, 1, 0, 1], [3, 4, 1, 0]], dtype=np.float32)
        idx, dst = rslib.knn(D, 2)
        self.assertEqual(idx[0].tolist(), [1, 2])
        g = np.array([0, 0, 1, 1])
        idx, _ = rslib.knn(D, 2, mask_group=g)
        self.assertEqual(idx[0].tolist()[:2], [2, 3])
        self.assertTrue((idx[0] != 1).all())
        idx, _ = rslib.knn(D, 2, cand=np.array([False, True, True, False]))
        self.assertEqual(sorted(idx[0].tolist()), [1, 2])

    def test_purity_chance(self):
        idx = np.array([[1], [0], [3], [2]])
        lab = np.array([0, 0, 1, 1])
        p, c = rslib.purity(idx, lab)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(c, 0.5)


if __name__ == "__main__":
    unittest.main()
