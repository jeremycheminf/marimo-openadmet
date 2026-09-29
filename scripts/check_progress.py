"""Live progress monitor for the buy-RTLogD sweeps (scripts 08 / 09).

    pixi run progress                 # one-shot report
    pixi run progress -- --watch 60   # refresh every 60s

Reports the running sweep process, how many runs are complete / remaining,
per-fraction breakdown, observed rate + ETA, and where the raw logs are.

The expected grid defaults mirror the commands the sweeps were launched with:

    ensemble : fractions 10..100, budgets 100/500/1000/5000,
               random x5 seeds + diversity_tanimoto + uncertainty
               + similarity                                    -> 320 runs
    chemprop : fractions 10/50/100, budgets 100/500/1000/5000,
               random x3 seeds + diversity_tanimoto + uncertainty
               + similarity                                    ->  72 runs

Override with --fractions / --budgets / --seeds / --config-seed if you launched
differently. A run counts as *complete* when its metrics CSV holds a baseline
row plus every buy round (same rule as ``reporting.run_complete``).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_METRICS = ROOT / "results" / "metrics"
LOG_FILES = {
    "ensemble": ROOT / "tmp" / "resume_ens_out.txt",
    "chemprop": ROOT / "tmp" / "resume_cp_out.txt",
}

DEFAULT_BUDGETS = [100, 500, 1000, 5000]
ENSEMBLE_FRACTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CHEMPROP_FRACTIONS = [10, 50, 100]
RANDOM_SEEDS = {"ensemble": 5, "chemprop": 3}
OTHER_STRATEGIES = ["diversity_tanimoto", "uncertainty", "similarity"]
CONFIG_SEED = 42


def n_rounds_for(purchase_step):
    """Mirrors active_learning.run_buy_simulation: n_rounds = round(1/purchase_step)."""
    return max(1, int(round(1.0 / purchase_step)))


def run_id(model, strategy, fraction_pct, seed, budget):
    return f"al_{model}_{strategy}_frac{fraction_pct:03d}_budget{budget:05d}_seed{seed}"


def metrics_path(model, strategy, fraction_pct, seed, budget):
    return RESULTS_METRICS / f"{run_id(model, strategy, fraction_pct, seed, budget)}_metrics.csv"


def run_state(model, strategy, fraction_pct, seed, budget, expected_rows):
    """'done' | 'partial' | 'missing'."""
    path = metrics_path(model, strategy, fraction_pct, seed, budget)
    if not path.exists():
        return "missing"
    try:
        with path.open(encoding="utf-8") as fh:
            rows = sum(1 for _ in fh) - 1
    except OSError:
        return "partial"
    return "done" if rows >= expected_rows else "partial"


def build_grid(fractions_cfg, seeds_cfg, budgets, config_seed):
    grid = []
    for model, fractions in fractions_cfg.items():
        for frac in fractions:
            for budget in budgets:
                for seed in range(seeds_cfg[model]):
                    grid.append((model, "random", frac, seed, budget))
                for strat in OTHER_STRATEGIES:
                    grid.append((model, strat, frac, config_seed, budget))
    return grid


def find_sweep_processes():
    """Return sweep python processes, or None if psutil is unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
        except Exception:
            continue
        joined = " ".join(cmdline)
        if "active_learning_buy" not in joined:
            continue
        if "python" not in (proc.info["name"] or "").lower():
            continue
        script = next((Path(c) for c in cmdline if "active_learning_buy" in c), None)
        try:
            start_time = proc.create_time()
            elapsed = time.time() - start_time
            cpu = proc.cpu_times().user + proc.cpu_times().system
            rss = proc.memory_info().rss
            alive = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            start_time = elapsed = cpu = rss = 0
            alive = True
        found.append(
            {
                "pid": proc.pid,
                "script": script.name if script else "?",
                "start_time": start_time,
                "elapsed": elapsed,
                "cpu": cpu,
                "rss": rss,
                "alive": alive,
            }
        )
    return found


def parse_log_progress(log_path):
    """Return (last_started, last_done, n_starts, n_dones) from a sweep stdout log."""
    if not log_path.exists():
        return None, None, 0, 0
    start_re = re.compile(r"\]\s+(\S+):\s*start")
    done_re = re.compile(r"\]\s+(\S+):\s*done\b")
    last_start = last_done = None
    n_starts = n_dones = 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None, 0, 0
    for line in text.splitlines():
        m = start_re.search(line)
        if m:
            last_start, n_starts = m.group(1), n_starts + 1
        m = done_re.search(line)
        if m:
            last_done, n_dones = m.group(1), n_dones + 1
    return last_start, last_done, n_starts, n_dones


def fmt_dur(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def bar(done, total, width=28):
    if total <= 0:
        return ""
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def report(grid, expected_rows, purchase_step):
    now = datetime.now()
    print("=" * 74)
    print(f"BUY-SWEEP PROGRESS  |  {now:%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    procs = find_sweep_processes()
    print("\nRUNNING PROCESS")
    if procs is None:
        print("  (psutil not installed - cannot inspect processes)")
    elif not procs:
        print("  no sweep process found  ->  the sweep is NOT running "
              "(stopped, finished, or still launching)")
    else:
        for p in procs:
            avg = (p["cpu"] / p["elapsed"] * 100) if p["elapsed"] else 0
            print(f"  PID {p['pid']:<7d} {p['script']}")
            print(
                f"      elapsed {fmt_dur(p['elapsed'])}   "
                f"cpu {fmt_dur(p['cpu'])} ({avg:.0f}%)   "
                f"rss {p['rss'] / 1e9:.2f} GB"
            )

    states = {key: run_state(*key, expected_rows) for key in grid}
    models = sorted({k[0] for k in grid})
    per_model = {}
    for model in models:
        keys = [k for k in grid if k[0] == model]
        done = sum(1 for k in keys if states[k] == "done")
        partial = sum(1 for k in keys if states[k] == "partial")
        per_model[model] = (done, len(keys), partial)

    total_done = sum(v[0] for v in per_model.values())
    total_n = sum(v[1] for v in per_model.values())

    print(f"\nPROGRESS  (a run = baseline row + all buy rounds; "
          f"purchase_step={purchase_step} -> {expected_rows} rows)")
    for model in models:
        done, n, partial = per_model[model]
        pct = 100.0 * done / n if n else 0.0
        extra = f"  ({partial} partial/in-flight)" if partial else ""
        print(f"  {model:<9s} {bar(done, n)} {done:3d}/{n:<3d} {pct:5.1f}%"
              f"   {n - done} left{extra}")
    pct = 100.0 * total_done / total_n if total_n else 0.0
    print(f"  {'TOTAL':<9s} {bar(total_done, total_n)} {total_done:3d}/{total_n:<3d} "
          f"{pct:5.1f}%   {total_n - total_done} left")

    for model in models:
        fractions = sorted({k[2] for k in grid if k[0] == model})
        print(f"\nPER FRACTION ({model})")
        print("   frac   done/total")
        for frac in fractions:
            keys = [k for k in grid if k[0] == model and k[2] == frac]
            done = sum(1 for k in keys if states[k] == "done")
            mark = "  done" if done == len(keys) else ("  not started" if done == 0 else "")
            print(f"   {frac:3d}%   {done:3d}/{len(keys):<3d}  "
                  f"{bar(done, len(keys), 20)}{mark}")

    print("\nRATE / ETA")
    mtimes = []
    for key in grid:
        if states[key] == "done":
            try:
                mtimes.append(metrics_path(*key).stat().st_mtime)
            except OSError:
                pass
    if len(mtimes) < 2:
        print("  not enough completed runs yet to estimate a rate")
    else:
        mtimes.sort()
        newest = mtimes[-1]
        rate = None

        window = [m for m in mtimes if newest - m <= 1800]
        if len(window) >= 2:
            span = max(window[-1] - window[0], 1e-6)
            recent = (len(window) - 1) / (span / 3600.0)
            print(f"  last 30m   : {len(window) - 1} runs  ->  {recent:.1f} runs/hour")
            rate = recent

        if procs:
            since = min(p["start_time"] for p in procs if p["start_time"])
            sess = [m for m in mtimes if m >= since]
            if len(sess) >= 2:
                span = max(sess[-1] - sess[0], 1e-6)
                rate = (len(sess) - 1) / (span / 3600.0)
                print(f"  this sweep : {len(sess)} runs since "
                      f"{datetime.fromtimestamp(since):%H:%M}  ->  {rate:.1f} runs/hour")
            stale = [m for m in mtimes if m < since]
            if stale:
                print(f"  (ignoring {len(stale)} run(s) from earlier sessions "
                      f"for the rate/ETA)")

        remaining = total_n - total_done
        if rate and remaining:
            eta_s = remaining / rate * 3600
            print(f"  remaining  : {remaining} runs  ->  ~{fmt_dur(eta_s)} at that rate")
            print("  (approximate - run cost grows with training size, so the "
                  "tail is slower)")
            print(f"  estimated completion: "
                  f"{datetime.fromtimestamp(newest + eta_s):%a %d %b %H:%M}")
        elif not remaining:
            print("  all runs complete")

    print("\nCURRENT ACTIVITY")
    newest_file = None
    for f in RESULTS_METRICS.glob("al_*_metrics.csv"):
        try:
            mt = f.stat().st_mtime
        except OSError:
            continue
        if newest_file is None or mt > newest_file[1]:
            newest_file = (f, mt)
    if newest_file:
        age = now.timestamp() - newest_file[1]
        warn = "   <-- STALLED?" if age > 900 and procs else ""
        print(f"  newest metrics file: {newest_file[0].name}")
        print(f"  written {fmt_dur(age)} ago{warn}")
    for model, log in LOG_FILES.items():
        last_start, last_done, n_starts, n_dones = parse_log_progress(log)
        if last_start is None:
            continue
        in_flight = last_start if last_start != last_done else "(none - between runs)"
        print(f"  {model:<9s} log: {n_dones}/{n_starts} started runs finished; "
              f"in flight: {in_flight}")

    print("\nRAW LOGS (live tail)")
    for model, log in LOG_FILES.items():
        try:
            rel = log.relative_to(ROOT)
        except ValueError:
            rel = log
        print(f"  # {model}:  Get-Content '{rel}' -Wait -Tail 20")
    print("  # count metric files:  "
          "(Get-ChildItem results\\metrics\\al_*_metrics.csv).Count")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(
        description="Live progress monitor for the buy-RTLogD sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fractions", default=None,
                    help="override fractions for BOTH models, e.g. '10,50,100'")
    ap.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS))
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the random-seed count for BOTH models")
    ap.add_argument("--config-seed", type=int, default=CONFIG_SEED)
    ap.add_argument("--purchase-step", type=float, default=0.25)
    ap.add_argument("--watch", type=float, default=0,
                    help="refresh every N seconds (0 = one-shot)")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    if args.fractions:
        fractions = [int(f) for f in args.fractions.split(",") if f.strip()]
        fractions_cfg = {"ensemble": fractions, "chemprop": fractions}
    else:
        fractions_cfg = {"ensemble": ENSEMBLE_FRACTIONS, "chemprop": CHEMPROP_FRACTIONS}
    seeds_cfg = dict(RANDOM_SEEDS)
    if args.seeds is not None:
        seeds_cfg = {m: args.seeds for m in seeds_cfg}

    grid = build_grid(fractions_cfg, seeds_cfg, budgets, args.config_seed)
    expected_rows = n_rounds_for(args.purchase_step) + 1

    if args.watch <= 0:
        report(grid, expected_rows, args.purchase_step)
        return 0
    while True:
        print("\033[H\033[J", end="")
        report(grid, expected_rows, args.purchase_step)
        print(f"\n(refreshing every {args.watch:g}s - Ctrl+C to stop)")
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print()
            return 0


if __name__ == "__main__":
    sys.exit(main())
