import pandas as pd
import numpy as np
import re
from scipy.stats import kendalltau, spearmanr
from typing import Dict, List, Tuple, Optional
import sys

# ─── Ground truth files (one per decision type) ──────────────────────────────
GROUND_TRUTH_CSVS = {
    'HVAC':      'ground_truth_hvac.csv',
    'Shower':    'ground_truth_shower.csv',
    'Appliance': 'ground_truth_appliance.csv',
}

ARCHITECTURE_CSVS = {
    'Pure':   'pure_prompting_results.csv',
    'RAG':    'RAGResults.csv',
    'Hybrid': 'hybrid_results.csv',
}

OUTPUT_CSV = 'metrics_comparison.csv'

CRITERIA = ['energy_cost', 'environmental', 'comfort', 'practicality']

# Scenario-ID ranges used in result files to infer decision_type when the
# result file has no decision_type column (e.g. pure_prompting_results.csv).
RESULT_DT_RANGES = {
    'HVAC':      (1,  58),
    'Appliance': (59, 84),
    'Shower':    (85, 99),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def norm_alt(x: str) -> str:
    """
    Normalise alternative labels so they can be used as join keys.

    Handles two quirks found in the data:
      • Temperature alternatives that appear as '72' in one file and '72F'/'72f'
        in another.
      • Float-formatted integers stored as '72.0'.
    Everything else is lowercased and stripped.
    """
    s = str(x).strip()
    s = re.sub(r'[Ff]$', '', s)          # strip trailing °F suffix
    try:
        return str(int(float(s)))         # '72.0' → '72'
    except ValueError:
        return s.lower()


def make_match_key(group_df: pd.DataFrame, alt_col: str = 'alternative') -> str:
    """
    Build a deterministic string key: '<location>|<sorted normalised alts>'.

    This is used as the join key between GT and prediction DataFrames instead
    of scenario_id, because the two sets of files use independent ID spaces.
    """
    location = group_df['location'].iloc[0]
    alts     = tuple(sorted(group_df[alt_col].apply(norm_alt)))
    return f"{location}|{alts}"


def add_decision_type_from_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX #6 – pure_prompting_results.csv has no decision_type column.
    Infer it from the global scenario_id ranges used when the dataset was built.
    """
    df = df.copy()
    df['decision_type'] = 'Unknown'
    for dt, (lo, hi) in RESULT_DT_RANGES.items():
        mask = (df['scenario_id'] >= lo) & (df['scenario_id'] <= hi)
        df.loc[mask, 'decision_type'] = dt
    return df


def load_ground_truth() -> Dict[str, pd.DataFrame]:
    """
    Load each GT file, apply all necessary fixes, and return a dict
    keyed by decision type.

    Fixes applied per file
    ──────────────────────
    FIX #1  column renamed from 'scenario_id' to 'scenario_id' (already ok),
            but 'scenario' does not exist – script previously used 'scenario'.
    FIX #2  GT files have no 'rank' column; it is computed here from a
            weighted mean of the four criteria (equal weights, rank 1 = best).
    FIX #3  Score columns carry a '_score' suffix (e.g. 'energy_cost_score')
            while the rest of the codebase expects bare names like 'energy_cost'.
    FIX #5  The HVAC file contains a NaN scenario_id row that is dropped.
    """
    print("Loading ground truth files…")
    gt_tables = {}

    for decision_type, csv_path in GROUND_TRUTH_CSVS.items():
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"  ❌  {csv_path} not found – skipping {decision_type}")
            continue

        # FIX #5 – drop rows with NaN scenario_id (found in HVAC file)
        before = len(df)
        df = df.dropna(subset=['scenario_id']).copy()
        if len(df) < before:
            print(f"  ⚠   Dropped {before - len(df)} NaN scenario_id rows from {csv_path}")

        # FIX #3 – rename '*_score' columns to bare criterion names
        rename_map = {f"{c}_score": c for c in CRITERIA if f"{c}_score" in df.columns}
        df = df.rename(columns=rename_map)

        # Validate required columns are now present
        missing = [c for c in ['scenario_id', 'location', 'alternative'] + CRITERIA
                   if c not in df.columns]
        if missing:
            print(f"  ❌  {csv_path} still missing columns after rename: {missing}")
            continue

        # FIX #2 – compute rank within each scenario (rank 1 = highest mean score)
        df['_weighted'] = df[CRITERIA].mean(axis=1)
        df['rank'] = (
            df.groupby('scenario_id')['_weighted']
              .rank(ascending=False, method='min')
              .astype(int)
        )
        df = df.drop(columns=['_weighted'])

        # Build a match key per scenario for later joining
        df['match_key'] = (
            df.groupby('scenario_id', group_keys=False)
              .apply(lambda g: pd.Series(make_match_key(g), index=g.index))
        )

        n_scen = df['scenario_id'].nunique()
        print(f"  ✓  {csv_path}: {len(df)} rows, {n_scen} scenarios")
        gt_tables[decision_type] = df

    return gt_tables


def load_prediction(csv_path: str, arch_name: str) -> Optional[pd.DataFrame]:
    """
    Load a prediction file, apply column fixes, and return a clean DataFrame.

    Fixes applied
    ─────────────
    FIX #1  'scenario_id' is present (not 'scenario') – validate and keep.
    FIX #6  If 'decision_type' is absent (Pure Prompting), infer from ID range.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ❌  {csv_path} not found – skipping {arch_name}")
        return None

    # FIX #1 – ensure we have scenario_id
    if 'scenario_id' not in df.columns:
        print(f"  ❌  {csv_path} has no 'scenario_id' column – skipping")
        return None

    # FIX #6 – add decision_type if missing
    if 'decision_type' not in df.columns:
        print(f"  ⚠   {csv_path} has no 'decision_type' column – inferring from ID range")
        df = add_decision_type_from_range(df)

    # Validate score columns
    missing = [c for c in CRITERIA + ['rank'] if c not in df.columns]
    if missing:
        print(f"  ❌  {csv_path} missing columns: {missing} – skipping")
        return None

    n_scen = df['scenario_id'].nunique()
    print(f"  ✓  {csv_path}: {len(df)} rows, {n_scen} scenarios")
    return df


# ─── Matching ────────────────────────────────────────────────────────────────

def build_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-scenario DataFrame with a 'match_key' column."""
    return (
        df.groupby('scenario_id', group_keys=False)
          .apply(lambda g: pd.Series({'match_key': make_match_key(g)},
                                     name=g['scenario_id'].iloc[0]))
          .reset_index()
          .rename(columns={'index': 'scenario_id'})
    )


# ─── Metric functions ─────────────────────────────────────────────────────────

def calculate_kendalls_tau(gt_ranks: List[int], pred_ranks: List[int]) -> float:
    tau, _ = kendalltau(gt_ranks, pred_ranks)
    return float(tau)


def calculate_spearmans_rho(gt_ranks: List[int], pred_ranks: List[int]) -> float:
    rho, _ = spearmanr(gt_ranks, pred_ranks)
    return float(rho)


def calculate_top1_match(gt_ranks: List[int], pred_ranks: List[int]) -> int:
    return 1 if gt_ranks.index(1) == pred_ranks.index(1) else 0


def calculate_top2_match(gt_ranks: List[int], pred_ranks: List[int]) -> int:
    pred_top1_pos = pred_ranks.index(1)
    return 1 if gt_ranks[pred_top1_pos] in [1, 2] else 0


def calculate_mae(gt_scores: List[float], pred_scores: List[float]) -> float:
    return float(np.mean([abs(p - g) for p, g in zip(pred_scores, gt_scores)]))


def calculate_rmse(gt_scores: List[float], pred_scores: List[float]) -> float:
    return float(np.sqrt(np.mean([(p - g) ** 2 for p, g in zip(pred_scores, gt_scores)])))


# ─── Per-scenario calculation ─────────────────────────────────────────────────

def calculate_scenario_metrics(gt_slice: pd.DataFrame,
                                pred_slice: pd.DataFrame) -> Dict:
    """
    Calculate all metrics for a matched GT / prediction pair.

    Both slices must already contain exactly 3 rows (one per alternative),
    sorted by the normalised alternative label so positions correspond.
    """
    gt_s   = gt_slice.copy()
    pred_s = pred_slice.copy()

    # Sort by normalised alternative so positions align
    gt_s['_alt_norm']   = gt_s['alternative'].apply(norm_alt)
    pred_s['_alt_norm'] = pred_s['alternative'].apply(norm_alt)
    gt_s   = gt_s.sort_values('_alt_norm').reset_index(drop=True)
    pred_s = pred_s.sort_values('_alt_norm').reset_index(drop=True)

    if len(gt_s) != 3 or len(pred_s) != 3:
        raise ValueError(f"Expected 3 alternatives each, got GT={len(gt_s)} Pred={len(pred_s)}")

    gt_ranks   = gt_s['rank'].tolist()
    pred_ranks = pred_s['rank'].tolist()

    results = {
        'kendalls_tau':  calculate_kendalls_tau(gt_ranks, pred_ranks),
        'spearmans_rho': calculate_spearmans_rho(gt_ranks, pred_ranks),
        'top1_match':    calculate_top1_match(gt_ranks, pred_ranks),
        'top2_match':    calculate_top2_match(gt_ranks, pred_ranks),
    }

    for criterion in CRITERIA:
        results[f'mae_{criterion}']  = calculate_mae(
            gt_s[criterion].tolist(), pred_s[criterion].tolist())
        results[f'rmse_{criterion}'] = calculate_rmse(
            gt_s[criterion].tolist(), pred_s[criterion].tolist())

    return results


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_metrics(metrics_df: pd.DataFrame, label: str) -> Dict:
    agg = {
        'scenario':      label,
        'kendalls_tau':  metrics_df['kendalls_tau'].mean(),
        'spearmans_rho': metrics_df['spearmans_rho'].mean(),
        'top1_match':    metrics_df['top1_match'].mean(),
        'top2_match':    metrics_df['top2_match'].mean(),
    }
    for criterion in CRITERIA:
        agg[f'mae_{criterion}']  = metrics_df[f'mae_{criterion}'].mean()
        agg[f'rmse_{criterion}'] = metrics_df[f'rmse_{criterion}'].mean()
    return agg


# ─── Main ─────────────────────────────────────────────────────────────────────

def calculate_all_metrics():
    print(f"\n{'=' * 70}")
    print("METRICS CALCULATOR  (fixed version)")
    print(f"{'=' * 70}\n")

    # FIX #3 + #2 + #5 applied inside load_ground_truth()
    gt_tables = load_ground_truth()
    if not gt_tables:
        print("❌  No ground truth loaded – aborting.")
        sys.exit(1)

    # Build per-scenario match-key lookup for each GT table
    # key: (decision_type, match_key) → (scenario_id, sorted GT slice)
    gt_key_index: Dict[Tuple[str, str], pd.DataFrame] = {}
    for dt, gt_df in gt_tables.items():
        for sid, g in gt_df.groupby('scenario_id'):
            key = g['match_key'].iloc[0]
            gt_key_index[(dt, key)] = g

    all_rows = []

    for arch_name, arch_csv in ARCHITECTURE_CSVS.items():
        print(f"\n{'-' * 70}")
        print(f"Processing: {arch_name}")
        print(f"{'-' * 70}")

        # FIX #1 + #6 applied inside load_prediction()
        pred_df = load_prediction(arch_csv, arch_name)
        if pred_df is None:
            continue

        scenario_metrics = []

        for dt in sorted(pred_df['decision_type'].unique()):
            if dt not in gt_tables:
                print(f"  ⚠   No GT table for decision_type '{dt}' – skipping")
                continue

            pred_dt = pred_df[pred_df['decision_type'] == dt]
            matched_count = 0
            skipped_count = 0

            for sid, pred_slice in pred_dt.groupby('scenario_id'):
                # FIX #4 – use content-based match key instead of scenario_id
                pred_key = make_match_key(pred_slice)
                gt_slice = gt_key_index.get((dt, pred_key))

                if gt_slice is None:
                    skipped_count += 1
                    continue

                try:
                    metrics = calculate_scenario_metrics(gt_slice, pred_slice)
                except Exception as e:
                    print(f"    ❌ scenario_id={sid} ({dt}): {e}")
                    skipped_count += 1
                    continue

                row = {
                    'architecture': arch_name,
                    'scenario':     pred_key,     # human-readable composite key
                    'scenario_id':  sid,
                    'decision_type': dt,
                    **metrics,
                }
                scenario_metrics.append(row)
                all_rows.append(row)
                matched_count += 1

            print(f"  {dt}: matched {matched_count} scenarios "
                  f"({skipped_count} skipped – no GT counterpart found)")

        if not scenario_metrics:
            print(f"  ⚠   No matchable scenarios for {arch_name} – skipping aggregation")
            continue

        scenario_df = pd.DataFrame(scenario_metrics)

        # Overall aggregation
        overall = aggregate_metrics(scenario_df, 'OVERALL_MEAN')
        overall['architecture']  = arch_name
        overall['scenario_id']   = pd.NA
        overall['decision_type'] = 'all'
        all_rows.append(overall)

        print(f"\n  Overall Kendall's tau : {overall['kendalls_tau']:.3f}")
        print(f"  Overall Spearman's rho: {overall['spearmans_rho']:.3f}")
        print(f"  Overall Top-1 accuracy: {overall['top1_match']:.1%}")
        print(f"  Overall Top-2 accuracy: {overall['top2_match']:.1%}")

        # Per-decision-type aggregation
        for dt in sorted(scenario_df['decision_type'].unique()):
            dt_df  = scenario_df[scenario_df['decision_type'] == dt]
            dt_agg = aggregate_metrics(dt_df, f'{dt}_MEAN')
            dt_agg['architecture']  = arch_name
            dt_agg['scenario_id']   = pd.NA
            dt_agg['decision_type'] = dt
            all_rows.append(dt_agg)
            print(f"  {dt:10s} – Kendall's tau: {dt_agg['kendalls_tau']:.3f}, "
                  f"Top-1: {dt_agg['top1_match']:.1%}")

    if not all_rows:
        print("\n❌  No results to save.")
        sys.exit(1)

    results_df = pd.DataFrame(all_rows)

    col_order = ['architecture', 'scenario', 'scenario_id', 'decision_type',
                 'kendalls_tau', 'spearmans_rho', 'top1_match', 'top2_match']
    for criterion in CRITERIA:
        col_order.append(f'mae_{criterion}')
    for criterion in CRITERIA:
        col_order.append(f'rmse_{criterion}')

    results_df = results_df[[c for c in col_order if c in results_df.columns]]
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 70}")
    print(f"RESULTS SAVED  →  {OUTPUT_CSV}  ({len(results_df)} rows)")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    calculate_all_metrics()