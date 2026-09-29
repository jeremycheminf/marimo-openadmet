"""Build the committed notebook assets (Phase 1 of the notebook rewrite).

    pixi run python scripts/10_build_notebook_assets.py
    pixi run python scripts/10_build_notebook_assets.py --skip-scaffolds

Writes `assets/` (~3 MB total, committable -- no 250 MB feature matrices, no
296-file glob at notebook load time):

    metrics_all.parquet        every run's per-round metrics, both models
    purchases_slim.parquet     purchase log, no SMILES (join on pool_index)
    map_coords_slim.parquet    UMAP coords, float32, no SMILES
    {pool,train,test}_smiles.parquet
    pool_props.parquet         maxTc-to-train, MW, cLogP, TPSA, HBD, aryl rings
    nn_tanimoto.parquet        test->pool, test->train, pool->train max Tc
    scaffolds.parquet          Bemis-Murcko scaffold per compound
    demo_pool.parquet          3,000 pool compounds stratified on maxTc deciles

Keyed on `model in {ensemble, chemprop}`, so re-running this picks up new
chemprop results with no notebook change. Reuses the naming contract in
reporting.py (run_id / collect_metrics) instead of re-deriving it.

It also prints a validation report with the numbers the notebook narrative cites
(maxTc distribution and the hard wall, per-strategy property profile, scaffold
overlap, tau decay), so the story is checked at build time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

from src.marimo_openadmet import reporting  # noqa: E402
from src.marimo_openadmet.active_learning import METRICS_COLUMNS, max_tanimoto_to_reference  # noqa: E402

ASSETS = ROOT / "assets"
PROCESSED = ROOT / "data" / "processed"
MODELS = ("ensemble", "chemprop")
SOURCES = ("pool", "train", "test")
TAU_STEPS = (0.3, 0.4, 0.5, 0.6, 0.7)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_csv_safe(path: Path):
    """Read a CSV, tolerating a file the running sweep is mid-write."""
    try:
        df = pd.read_csv(path)
        return df if len(df) else None
    except Exception as exc:  # noqa: BLE001 - report and move on
        log(f"  !! skipping unreadable {path.name}: {type(exc).__name__}: {exc}")
        return None


def processed_path(source: str) -> Path:
    name = (
        f"openadmet_{source}_std_df.parquet"
        if source in ("train", "test")
        else f"rtlogd_{source}_std_df.parquet"
    )
    return PROCESSED / name


def build_metrics_all() -> pd.DataFrame:
    parts = []
    for model in MODELS:
        files = sorted(reporting.RESULT_METRICS.glob(f"al_{model}_*_metrics.csv"))
        if not files:
            log(f"metrics: no files for '{model}' yet")
            continue
        frames = [d for f in files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True).reindex(columns=METRICS_COLUMNS)
        # purchase_pct is a percentage *of the budget*, not of the train set:
        # alias it unambiguously so charts cannot mislabel it.
        df["pct_of_budget"] = df["purchase_pct"]
        parts.append(df)
        log(f"metrics: {model}: {len(files)} files -> {len(df)} rows")
    if not parts:
        log("metrics: nothing found")
        return pd.DataFrame(columns=[*METRICS_COLUMNS, "pct_of_budget"])
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(ASSETS / "metrics_all.parquet", index=False)
    log(f"metrics_all.parquet: {out.shape}, {out['model'].nunique()} model(s)")
    return out


def build_purchases_slim() -> pd.DataFrame:
    parts = []
    for model in MODELS:
        files = sorted(reporting.RESULT_METRICS.glob(f"al_{model}_*_selected.csv"))
        if not files:
            continue
        frames = [d for f in files if (d := _read_csv_safe(f)) is not None]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df.insert(0, "model", model)
        parts.append(df)
        log(f"purchases: {model}: {len(files)} files -> {len(df)} rows")
    if not parts:
        log("purchases: nothing found")
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop(columns=["smiles"])  # join on pool_index instead
    for col in ("model", "strategy"):
        out[col] = out[col].astype("category")
    for col in ("seed", "round", "rank_in_round", "pool_index"):
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in ("fraction", "purchase_pct", "oracle_logd", "acquisition_score"):
        out[col] = out[col].astype("float32")
    out.to_parquet(ASSETS / "purchases_slim.parquet", index=False)
    log(f"purchases_slim.parquet: {out.shape}")
    return out


def build_map_slim() -> pd.DataFrame:
    src = reporting.RESULT_METRICS / "al_map_coords.parquet"
    if not src.exists():
        log("map: cache missing (run script 08 with --prepare-map-only)")
        return pd.DataFrame()
    df = pd.read_parquet(src)
    smiles_col = next((c for c in ("smiles", "SMILES") if c in df.columns), None)
    if smiles_col:
        df = df.drop(columns=[smiles_col])
    for col in ("umap_x", "umap_y", "logd"):
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df["source"] = df["source"].astype("category")
    df.to_parquet(ASSETS / "map_coords_slim.parquet", index=False)
    log(f"map_coords_slim.parquet: {df.shape}, sources={df['source'].value_counts().to_dict()}")
    return df


def build_smiles() -> dict:
    out = {}
    for source in SOURCES:
        src = processed_path(source)
        if not src.exists():
            log(f"smiles: missing {src.name}")
            continue
        df = pd.read_parquet(src, columns=["SMILES", "LogD"])
        df = df.rename(columns={"SMILES": "smiles", "LogD": "logd"})
        df.insert(0, "source_index", np.arange(len(df), dtype=np.int32))
        df["source"] = source
        df.to_parquet(ASSETS / f"{source}_smiles.parquet", index=False)
        out[source] = df
        log(f"{source}_smiles.parquet: {df.shape}")
    return out


def compute_props(smiles: list, label: str = "") -> pd.DataFrame:
    """MW, cLogP, TPSA, HBD, aromatic rings for a list of SMILES (NaN on invalid)."""
    rows = []
    n_bad = 0
    t0 = time.time()
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            n_bad += 1
            rows.append((np.nan,) * 5)
        else:
            rows.append(
                (
                    Descriptors.MolWt(mol),
                    Crippen.MolLogP(mol),
                    rdMolDescriptors.CalcTPSA(mol),
                    Lipinski.NumHDonors(mol),
                    rdMolDescriptors.CalcNumAromaticRings(mol),
                )
            )
        if (i + 1) % 4000 == 0:
            log(f"  props {label}: {i + 1}/{len(smiles)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(
        rows, columns=["mw", "clogp", "tpsa", "hbd", "aromatic_rings"], dtype="float32"
    )
    if n_bad:
        log(f"  props {label}: {n_bad} unparseable SMILES -> NaN properties")
    return df


def build_pool_props(pool_df: pd.DataFrame, train_smiles: list) -> pd.DataFrame:
    log(f"pool props: {len(pool_df)} compounds (RDKit descriptors)")
    df = compute_props(list(pool_df["smiles"].values), "pool")
    log("pool props: max Tanimoto vs full OpenADMET train (this is the slow one)...")
    t0 = time.time()
    _tc, _nn = max_tanimoto_to_reference(
        pool_df["smiles"].values, train_smiles, return_index=True
    )
    df["max_tc_vs_full_train"] = _tc.astype("float32")
    # index into the train_smiles list, for the notebook's detail card
    df["nn_train_index"] = _nn.astype("int32")
    log(f"  maxTc vs train done in {time.time() - t0:.0f}s")
    df.insert(0, "pool_index", np.arange(len(pool_df), dtype=np.int32))
    df.insert(1, "smiles", pool_df["smiles"].values)
    df["logd"] = pool_df["logd"].values.astype("float32")
    df.to_parquet(ASSETS / "pool_props.parquet", index=False)
    log(f"pool_props.parquet: {df.shape}")
    return df


def build_nn_tanimoto(smiles_by_source: dict) -> pd.DataFrame:
    """Nearest-neighbour Tanimoto, long format.

    The three query/reference pairs have different query lengths
    (test->pool: 2155, test->train: 2155, pool->train: 21278), so a wide
    rectangle is impossible. One row per (query_source, query_index,
    ref_source) with its max Tanimoto; pivot per (query_source, ref_source)
    in the notebook.
    """
    pool_sm = [str(s) for s in smiles_by_source["pool"]["smiles"].values]
    train_sm = [str(s) for s in smiles_by_source["train"]["smiles"].values]
    test_sm = [str(s) for s in smiles_by_source["test"]["smiles"].values]
    log(f"nn_tanimoto: test->{len(test_sm)}, pool->{len(pool_sm)} vs train {len(train_sm)}")
    t0 = time.time()

    def nn(query_smiles: list, ref_smiles: list) -> np.ndarray:
        return np.asarray(
            max_tanimoto_to_reference(query_smiles, ref_smiles), dtype="float32"
        )

    test_to_pool = nn(test_sm, pool_sm)
    test_to_train = nn(test_sm, train_sm)
    pool_to_train = nn(pool_sm, train_sm)
    assert len(test_to_pool) == len(test_sm) == len(test_to_train), (
        len(test_to_pool),
        len(test_sm),
        len(test_to_train),
    )
    assert len(pool_to_train) == len(pool_sm), (
        len(pool_to_train),
        len(pool_sm),
    )

    out = pd.DataFrame(
        {
            "query_source": pd.Categorical(
                ["test"] * (2 * len(test_sm)) + ["pool"] * len(pool_sm),
                categories=["test", "pool"],
            ),
            "query_index": np.concatenate(
                [
                    np.arange(len(test_sm), dtype="int32"),
                    np.arange(len(test_sm), dtype="int32"),
                    np.arange(len(pool_sm), dtype="int32"),
                ]
            ),
            "ref_source": pd.Categorical(
                ["pool"] * len(test_sm)
                + ["train"] * len(test_sm)
                + ["train"] * len(pool_sm),
                categories=["pool", "train"],
            ),
            "max_tc": np.concatenate(
                [test_to_pool, test_to_train, pool_to_train]
            ).astype("float32"),
        }
    )
    assert len(out) == 2 * len(test_sm) + len(pool_sm), len(out)
    out.to_parquet(ASSETS / "nn_tanimoto.parquet", index=False)
    log(f"nn_tanimoto.parquet: {out.shape} ({time.time() - t0:.0f}s)")
    return out


def build_scaffolds(smiles_by_source: dict, skip: bool = False) -> pd.DataFrame:
    rows = []
    if skip:
        log("scaffolds: skipped (--skip-scaffolds)")
        return pd.DataFrame(columns=["source", "source_index", "smiles", "scaffold"])
    for source in SOURCES:
        df = smiles_by_source[source]
        t0 = time.time()
        for i, s in enumerate(df["smiles"].values):
            try:
                scf = MurckoScaffold.MurckoScaffoldSmiles(smiles=str(s))
            except Exception:  # noqa: BLE001
                scf = ""
            rows.append((source, int(df["source_index"].iloc[i]), s, scf))
            if (i + 1) % 4000 == 0:
                log(f"  scaffolds {source}: {i + 1}/{len(df)} ({time.time() - t0:.0f}s)")
    out = pd.DataFrame(rows, columns=["source", "source_index", "smiles", "scaffold"])
    out.to_parquet(ASSETS / "scaffolds.parquet", index=False)
    log(f"scaffolds.parquet: {out.shape}")
    return out


def build_demo_pool(pool_props: pd.DataFrame, n_demo: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Stratified sample across maxTc deciles so the live lab sees the whole range."""
    tc = pool_props["max_tc_vs_full_train"].values
    ranks = pd.Series(tc).rank(method="first")
    decile = pd.qcut(ranks, q=10, labels=False, duplicates="drop")
    pool_props = pool_props.assign(_decile=decile)
    per_bin = max(1, n_demo // pool_props["_decile"].nunique())
    rng = np.random.default_rng(seed)
    picks = []
    for b, grp in pool_props.groupby("_decile"):
        take = min(per_bin, len(grp))
        picks.append(grp.sample(n=take, random_state=rng.integers(1 << 31)))
    demo = (
        pd.concat(picks, ignore_index=True)
        .sort_values("max_tc_vs_full_train", ascending=False)
        .reset_index(drop=True)
    )
    demo = demo.drop(columns=["_decile"])
    demo.to_parquet(ASSETS / "demo_pool.parquet", index=False)
    log(f"demo_pool.parquet: {demo.shape} (stratified on maxTc deciles, seed={seed})")
    return demo


def validation_report(
    pool_props: pd.DataFrame,
    nn: pd.DataFrame,
    scaffolds: pd.DataFrame,
    purchases: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    print("\n" + "=" * 74)
    print("VALIDATION REPORT -- numbers the notebook narrative cites")
    print("=" * 74)

    tc = pool_props["max_tc_vs_full_train"].values
    print(f"\n[pool -> full OpenADMET train] n={len(tc)}")
    print(f"  maxTc  mean={tc.mean():.3f}  median={np.median(tc):.3f}  "
          f"min={tc.min():.3f}  max={tc.max():.3f}")
    for thr in TAU_STEPS:
        print(f"  >= {thr:.1f}: {int((tc >= thr).sum()):5d} compounds")
    print("  ^ the tau-decay curve: eligibility collapses to 0 by tau=0.7")

    if not nn.empty:
        print("\n[nearest-neighbour Tanimoto]")
        for qs, rs in (("test", "pool"), ("test", "train"), ("pool", "train")):
            v = nn.loc[
                (nn["query_source"] == qs) & (nn["ref_source"] == rs), "max_tc"
            ].to_numpy(dtype="float64")
            print(
                f"  {qs + '->' + rs:<12s} n={len(v):5d} "
                f"mean={np.nanmean(v):.3f} median={np.nanmedian(v):.3f}"
            )

    if not purchases.empty and not pool_props.empty:
        print("\n[property profile of what each strategy bought "
              "(finding 4: diversity must NOT be a size maximiser)]")
        prof = (
            purchases.assign(mw=purchases["pool_index"].map(
                pd.Series(pool_props["mw"].values, index=pool_props["pool_index"].values)))
            .assign(clogp=purchases["pool_index"].map(
                pd.Series(pool_props["clogp"].values, index=pool_props["pool_index"].values)))
        )
        agg = prof.groupby(["model", "strategy"], observed=True).agg(
            n=("pool_index", "size"), mw=("mw", "mean"), clogp=("clogp", "mean")
        )
        print(agg.round(2).to_string())
        print(f"  pool reference: mw={pool_props['mw'].mean():.0f} "
              f"clogp={pool_props['clogp'].mean():.2f}")

    if not scaffolds.empty and {"test", "pool"} <= set(scaffolds["source"].unique()):
        t_sc = set(scaffolds.loc[scaffolds["source"] == "test", "scaffold"]) - {""}
        p_sc = set(scaffolds.loc[scaffolds["source"] == "pool", "scaffold"]) - {""}
        tr_sc = set(scaffolds.loc[scaffolds["source"] == "train", "scaffold"]) - {""}
        print("\n[scaffolds (Bemis-Murcko)]")
        print(f"  unique test scaffolds   : {len(t_sc)}")
        print(f"  ...also present in pool : {len(t_sc & p_sc)}")
        print(f"  ...also present in train: {len(t_sc & tr_sc)}")
        print("  ^ Act V: a handful of test scaffolds have a pool counterpart; "
              "the rest are grey.")

    if not metrics.empty:
        final = metrics.loc[
            metrics.groupby(["model", "strategy", "fraction", "budget", "seed"])["round"].idxmax()
        ]
        print("\n[end-state RMSE by model x strategy x budget (mean over seeds)]")
        agg = (
            final.groupby(["model", "strategy", "budget"], observed=True)["rmse"]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        print(agg.to_string())
    print("=" * 74)


def build_test_predictions() -> pd.DataFrame:
    """Final-round test predictions per run (script 15), packed for the parity plot."""
    files = sorted(reporting.RESULT_METRICS.glob("test_predictions_*.parquet"))
    if not files:
        log("test_predictions: none found (run scripts/15_test_predictions.py)")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[["model", "strategy", "fraction", "budget", "test_index", "pred", "n_train"]].copy()
    for c in ("model", "strategy"):
        df[c] = df[c].astype("category")
    df["fraction"] = df["fraction"].round(2).astype("float32")
    df["budget"] = df["budget"].astype("int16")
    df["test_index"] = df["test_index"].astype("int16")
    df["n_train"] = df["n_train"].astype("int32")
    df["pred"] = df["pred"].round(3).astype("float32")
    df.to_parquet(ASSETS / "test_predictions.parquet", index=False)
    n_runs = df.groupby(["model", "strategy", "fraction", "budget"], observed=True).ngroups
    log(f"test_predictions: {n_runs} models x {df['test_index'].nunique()} test compounds")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo-n", type=int, default=3000)
    ap.add_argument("--skip-scaffolds", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    log(f"assets -> {ASSETS}")

    metrics = build_metrics_all()
    purchases = build_purchases_slim()
    build_map_slim()
    smiles_by_source = build_smiles()
    if not all(s in smiles_by_source for s in SOURCES):
        log("FATAL: missing a processed parquet; cannot build chemistry assets")
        return 1

    pool_props = build_pool_props(
        smiles_by_source["pool"], list(smiles_by_source["train"]["smiles"].values)
    )
    nn = build_nn_tanimoto(smiles_by_source)
    scaffolds = build_scaffolds(smiles_by_source, skip=args.skip_scaffolds)
    build_demo_pool(pool_props, n_demo=args.demo_n)
    build_test_predictions()

    if not args.no_report:
        validation_report(pool_props, nn, scaffolds, purchases, metrics)

    sizes = sorted(
        (f.name, f.stat().st_size) for f in ASSETS.glob("*.parquet")
    )
    total = sum(s for _, s in sizes)
    print("\nASSETS")
    for name, size in sizes:
        print(f"  {name:<28s} {size / 1024:8.1f} KB")
    print(f"  {'TOTAL':<28s} {total / 1024 / 1024:8.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
