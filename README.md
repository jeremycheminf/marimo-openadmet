# Does buying more compounds help your model?

Entry for [molab Notebook Competition #3](https://marimo.io/pages/events/notebook-competition-3)
("Bring Cheminformatics to Life", ADMET).

You have an in-house LogD dataset and a model built on it. A vendor offers
21,278 public LogD measurements. Should you buy — and does it matter which
compounds, or which model you train? The notebook answers along three axes:
the **stage** of the project (how much internal data you already hold), the
**acquisition strategy** (random, uncertainty, diversity, similarity), and the
**model** (tree ensemble vs graph neural network). The precomputed sweep is
392 buy simulations; the live section lets you run your own round in seconds.

Case study: OpenADMET ExpansionRx LogD (4,999 train / 2,155 test) buying from
an external pool, from the RTLogD repository (<https://github.com/myzhengSIMM/RTlogD/tree/main/>)

## The notebook

`notebook_slim.py` is the submission: an intro, seven sections and a verdict,
with the machinery cells below the story and their code hidden. It is
self-contained and reads the parquet files
in `assets/` — locally from disk, on molab over raw GitHub URLs. Nothing trains
on load; the only compute is the live buy round behind a button.

```bash
pixi run notebook          # marimo edit notebook_slim.py -- the editor
pixi run view              # marimo run  notebook_slim.py -- read-only app view, no code shown
uvx --from marimo marimo edit --sandbox notebook_slim.py   # clean env built from the PEP 723 header
pixi run marimo export html notebook_slim.py -o out.html --no-sandbox   # static HTML (add --no-include-code to strip source)
```

Cells display in file order while executing by dependency, so the story cells
sit at the top of the file and the engine below. Every cell is
`@app.cell(hide_code=True)` except the live-lab engine; in the editor a hidden
cell shows only its output plus a small "show code" toggle. The page width is
set on `marimo.App(width="medium")` — change it to `"full"` there.

A guided tour (wigglystuff `CellTour`) sits under the intro: ten cards, one per step of the
argument, each anchored to a named cell (`def tour_hook(...)` etc.). Renaming one of those cells
breaks its step, so keep the names. The tour is also the script for the video walkthrough.

## Reproducing the data

Everything under `assets/` (6.7 MB, committed) comes from this chain. Steps
01–03 download and standardise; 08/09 are the sweeps (hours); 06/11/13/12 are
the model-capacity study; 10 packs the results into the notebook assets.

```bash
pixi run python scripts/01_fetch_openadmet.py
pixi run python scripts/02_fetch_and_standardize_rtlogd.py
pixi run python scripts/03_featurise.py                         # ECFP + RDKit descriptors -> data/processed

pixi run python scripts/08_active_learning_buy_ensemble.py      # RefinementStack, 10 stages x 4 budgets
pixi run -e gpu python scripts/09_active_learning_buy_chemprop.py   # chemprop, 3 stages, 20 epochs (GPU env)
pixi run progress                                               # done / left / ETA while a sweep runs

pixi run -e gpu python scripts/06_baseline_chemprop.py          # internal / pool / both at 5 epochs (no flags: starts training immediately)
pixi run -e gpu python scripts/11_chemprop_epoch_control.py     # internal-only at matched gradient updates
pixi run -e gpu python scripts/13_chemprop_combined_converged.py    # both at 27 epochs
pixi run python scripts/12_model_capacity.py                    # -> assets/model_capacity.parquet

pixi run python scripts/15_test_predictions.py --ensemble        # parity-plot predictions (refits, ~2.5 h CPU)
pixi run -e gpu python scripts/15_test_predictions.py --chemprop   # reads chemprop preds + 3 GPU baselines
pixi run build-assets                                           # -> assets/*.parquet
```

Both sweep scripts take `--skip-existing` to resume. Defaults reproduce the
shipped grid: strategies `random,diversity_tanimoto,uncertainty,similarity`,
budgets `100,500,1000,5000`; `random` runs 5 seeds (ensemble) / 3 (chemprop),
the other strategies once. The `gpu` pixi environment exists only for the
offline chemprop runs; the notebooks never need it.

## Layout

| Path | Contents |
| ---- | -------- |
| `notebook_slim.py` | The notebooks (see above) |
| `assets/` | Committed parquet files the notebooks load |
| `scripts/01..15` | The reproduction chain, in order |
| `scripts/check_progress.py` | Sweep monitor (`pixi run progress`) |
| `src/marimo_openadmet/` | Engine: `active_learning.py` (buy simulation), `selectors.py`, `models.py` (RefinementStack, ChempropModel), `features.py`, `reporting.py` |
| `data/`, `models/`, `results/`, `logs/` | Gitignored: raw/processed data, checkpoints, per-run CSVs |

## AI disclosure

Built with AI assistance (Claude and DeepSeek Flash v4.1 - Cline free);
the disclosure in the notebook's closing
section describes what was generated and what was written and checked by hand.

## Licence

Data: OpenADMET (ExpansionRx challenge) and RTLogD under their respective terms.
Code and notebook: MIT (see `LICENSE`).
