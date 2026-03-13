#!/usr/bin/env python3
"""
calculate_metrics.py - Metrics evaluation for MCDA architecture comparison
Science Fair Project: LLM-assisted MCDA for Household Emissions Optimization

Compares Pure Prompting, RAG-Enhanced, and Hybrid architectures against
physics-based ground truth using MAVT scoring across HVAC, Appliance,
and Shower decision scenarios.
"""

import pandas as pd
import numpy as np
from scipy import stats
import re
import warnings
import sys
from collections import defaultdict

warnings.filterwarnings("ignore")
CONFIG = {
    "ground_truth": {
        "HVAC": "ground_truth_hvac.csv",
        "Appliance": "ground_truth_appliance.csv",
        "Shower": "ground_truth_shower.csv",
    },
    "architectures": {
        "Pure": "pure_prompting_results.csv",
        "RAG": "RAGResults.csv",
        "Hybrid": "hybrid_results.csv",
    },
    "output_csv": "metrics_summary.csv",
    "weights": {
        "environmental": 0.35,
        "energy_cost": 0.30,
        "comfort": 0.20,
        "practicality": 0.15,
    },
    "gt_score_cols": {
        "energy_cost": "energy_cost_score",
        "environmental": "environmental_score",
        "comfort": "comfort_score",
        "practicality": "practicality_score",
    },
    "arch_score_cols": {
        "energy_cost": "energy_cost",
        "environmental": "environmental",
        "comfort": "comfort",
        "practicality": "practicality",
    },
}

CRITERIA = ["energy_cost", "environmental", "comfort", "practicality"]


def extract_time_from_alt(alt_str):
    """Extract time pattern from alternative string.
    Handles: '2:00 PM', 'Run dishwasher at 2:00 PM', '4PM', '1AM'."""
    alt_str = str(alt_str).strip()
    # Try full format first: '2:00 PM'
    match = re.search(r'(\d{1,2}:\d{2}\s*[AaPp][Mm])', alt_str)
    if match:
        return match.group(1).strip().upper()
    # Try abbreviated: '4PM', '1AM'\
    match = re.search(r'(\d{1,2})\s*([AaPp][Mm])', alt_str)
    if match:
        hour = match.group(1)
        ampm = match.group(2).upper()
        return f"{hour}:00 {ampm}"
    return alt_str.strip().upper()
def normalize_alternative(alt, decision_type):
    """Normalize alternative values for cross-file matching."""
    alt = str(alt).strip()
    if decision_type == "Appliance":
        return extract_time_from_alt(alt)
    else:
        try:
            return str(int(float(alt)))
        except ValueError:
            return alt


def load_ground_truth(config):
    """Load GT files separately by decision type (IDs overlap across types)."""
    gt_by_type = {}

    for dtype, filepath in config["ground_truth"].items():
        df = pd.read_csv(filepath)
        df["decision_type"] = dtype

        if "description" in df.columns and "question" not in df.columns:
            df = df.rename(columns={"description": "question"})

        rename_map = {}
        for criterion, gt_col in config["gt_score_cols"].items():
            if gt_col in df.columns:
                rename_map[gt_col] = f"gt_{criterion}"
        df = df.rename(columns=rename_map)

        if "rank" in df.columns:
            df = df.rename(columns={"rank": "gt_rank"})
        if "mavt_score" in df.columns:
            df = df.rename(columns={"mavt_score": "gt_mavt_score"})

        df["question"] = df["question"].str.strip()
        df["location"] = df["location"].str.strip()
        df["alternative"] = df["alternative"].astype(str).str.strip()

        gt_by_type[dtype] = df

    return gt_by_type

def load_architecture(filepath, arch_name):
    """Load an architecture results file."""
    df = pd.read_csv(filepath)
    df["architecture"] = arch_name
    df["question"] = df["question"].str.strip()
    df["location"] = df["location"].str.strip()
    df["alternative"] = df["alternative"].astype(str).str.strip()

    rename_map = {}
    for criterion in CRITERIA:
        col = CONFIG["arch_score_cols"][criterion]
        if col in df.columns:
            rename_map[col] = f"arch_{criterion}"
    df = df.rename(columns=rename_map)

    if "rank" in df.columns:
        df = df.rename(columns={"rank": "arch_rank"})
    if "weighted_score" in df.columns:
        df = df.rename(columns={"weighted_score": "arch_weighted_score"})

    return df
def build_gt_lookup(gt_by_type):
    """Build lookup: (question, location):   list of GT scenario entries."""
    gt_lookup = defaultdict(list)

    for dtype, gt_df in gt_by_type.items():
        for sid in gt_df["scenario_id"].unique():
            sub = gt_df[gt_df["scenario_id"] == sid]
            q = sub["question"].iloc[0]
            loc = sub["location"].iloc[0]

            alt_map = {}
            for _, row in sub.iterrows():
                norm_alt = normalize_alternative(row["alternative"], dtype)
                alt_map[norm_alt] = row

            gt_lookup[(q, loc)].append({
                "gt_sid": sid,
                "decision_type": dtype,
                "alt_map": alt_map,
                "used": False,
            })

    return gt_lookup


def match_scenarios(gt_lookup, arch_df, arch_name):
    """Match architecture scenarios to GT by question+location, then alternatives."""
    matched_rows = []
    warnings_log = []

    for arch_sid in arch_df["scenario_id"].unique():
        arch_sub = arch_df[arch_df["scenario_id"] == arch_sid]
        arch_dtype = arch_sub["decision_type"].iloc[0]
        q = arch_sub["question"].iloc[0]
        loc = arch_sub["location"].iloc[0]

        key = (q, loc)
        if key not in gt_lookup:
            warnings_log.append(
                f"No GT match: sid={arch_sid} ({arch_dtype}, '{q[:50]}', '{loc}')"
            )
            continue

        # Normalize arch alternatives
        arch_norm_alts = {}
        for _, row in arch_sub.iterrows():
            norm_alt = normalize_alternative(row["alternative"], arch_dtype)
            arch_norm_alts[norm_alt] = row

        # Find best GT entry: must match decision type
        best_match = None
        best_overlap = -1
        for gt_entry in gt_lookup[key]:
            if gt_entry["used"]:
                continue
            if gt_entry["decision_type"] != arch_dtype:
                continue
            overlap = len(set(gt_entry["alt_map"].keys()) & set(arch_norm_alts.keys()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = gt_entry

        if best_match is None or best_overlap == 0:
            warnings_log.append(
                f"No alt overlap: sid={arch_sid} ({arch_dtype}, '{q[:50]}', "
                f"arch_alts={list(arch_norm_alts.keys())})"
            )
            continue

        best_match["used"] = True

        for norm_alt, arch_row in arch_norm_alts.items():
            if norm_alt in best_match["alt_map"]:
                gt_row = best_match["alt_map"][norm_alt]

                merged = {
                    "arch_scenario_id": arch_sid,
                    "gt_scenario_id": best_match["gt_sid"],
                    "decision_type": arch_dtype,
                    "alternative": arch_row["alternative"],
                    "norm_alternative": norm_alt,
                    "architecture": arch_name,
                    "question": q,
                    "location": loc,
                }

                for c in CRITERIA:
                    merged[f"gt_{c}"] = gt_row.get(f"gt_{c}", np.nan)
                    merged[f"arch_{c}"] = arch_row.get(f"arch_{c}", np.nan)

                merged["gt_rank"] = gt_row.get("gt_rank", np.nan)
                merged["arch_rank"] = arch_row.get("arch_rank", np.nan)

                if "gt_mavt_score" in gt_row.index:
                    merged["gt_mavt_score"] = gt_row["gt_mavt_score"]
                if "arch_weighted_score" in arch_row.index:
                    merged["arch_weighted_score"] = arch_row["arch_weighted_score"]

                if "extraction_failed" in arch_row.index:
                    merged["extraction_failed"] = arch_row["extraction_failed"]
                if "gt_calculation_failed" in arch_row.index:
                    merged["gt_calculation_failed"] = arch_row["gt_calculation_failed"]

                matched_rows.append(merged)
            else:
                warnings_log.append(
                    f"Alt not in GT: sid={arch_sid}, alt='{norm_alt}' "
                    f"(GT has: {list(best_match['alt_map'].keys())})"
                )

    # Reset used flags for next architecture
    for entries in gt_lookup.values():
        for e in entries:
            e["used"] = False

    merged_df = pd.DataFrame(matched_rows)
    n_arch = arch_df["scenario_id"].nunique()
    n_matched = merged_df["arch_scenario_id"].nunique() if len(merged_df) > 0 else 0

    print(f"\n  [{arch_name}] Matched {n_matched}/{n_arch} scenarios "
          f"({len(merged_df)} alt rows)")

    if n_matched < n_arch:
        for dtype in ["HVAC", "Appliance", "Shower"]:
            arch_sids = set(arch_df[arch_df["decision_type"] == dtype]["scenario_id"].unique())
            matched_sids = set(
                merged_df[merged_df["decision_type"] == dtype]["arch_scenario_id"].unique()
            ) if len(merged_df) > 0 else set()
            unmatched = arch_sids - matched_sids
            if unmatched:
                print(f"    {dtype}: {len(matched_sids)}/{len(arch_sids)} matched, "
                      f"{len(unmatched)} missing")

    if warnings_log:
        n_show = min(5, len(warnings_log))
        print(f"    ({len(warnings_log)} warnings, showing {n_show})")
        for w in warnings_log[:n_show]:
            print(f"      {w}")

    return merged_df


def compute_criterion_metrics(merged_df):
    """Compute MAE and RMSE for each criterion and overall."""
    results = {}
    all_abs_errors = []
    all_sq_errors = []

    for c in CRITERIA:
        gt = merged_df[f"gt_{c}"].astype(float)
        arch = merged_df[f"arch_{c}"].astype(float)
        ae = (arch - gt).abs()
        se = (arch - gt) ** 2

        results[f"{c}_MAE"] = round(ae.mean(), 4)
        results[f"{c}_RMSE"] = round(np.sqrt(se.mean()), 4)

        all_abs_errors.extend(ae.tolist())
        all_sq_errors.extend(se.tolist())

    results["overall_MAE"] = round(np.mean(all_abs_errors), 4)
    results["overall_RMSE"] = round(np.sqrt(np.mean(all_sq_errors)), 4)
    return results


def compute_ranking_metrics(merged_df):
    """Kendall tau, Spearman rho, Top-1/Top-2 — per-scenario then averaged."""
    taus, rhos = [], []
    top1_ok = top2_ok = 0
    n = 0

    for sid in merged_df["arch_scenario_id"].unique():
        sc = merged_df[merged_df["arch_scenario_id"] == sid].copy()
        if len(sc) < 2:
            continue

        gt_r = sc["gt_rank"].astype(float).values
        ar_r = sc["arch_rank"].astype(float).values
        n += 1

        # Kendall
        if len(set(gt_r)) > 1 and len(set(ar_r)) > 1:
            tau, _ = stats.kendalltau(gt_r, ar_r)
            taus.append(tau if not np.isnan(tau) else 0.0)
        else:
            taus.append(1.0 if np.array_equal(gt_r, ar_r) else 0.0)

        # Spearman
        if len(set(gt_r)) > 1 and len(set(ar_r)) > 1:
            rho, _ = stats.spearmanr(gt_r, ar_r)
            rhos.append(rho if not np.isnan(rho) else 0.0)
        else:
            rhos.append(1.0 if np.array_equal(gt_r, ar_r) else 0.0)

        # top-1
        gt_top1 = sc.loc[sc["gt_rank"].astype(float).idxmin(), "norm_alternative"]
        ar_top1 = sc.loc[sc["arch_rank"].astype(float).idxmin(), "norm_alternative"]
        if gt_top1 == ar_top1:
            top1_ok += 1
        # top-2
        ar_top2 = set(sc.sort_values("arch_rank").head(2)["norm_alternative"])
        if gt_top1 in ar_top2:
            top2_ok += 1

    return {
        "kendall_tau": round(np.mean(taus), 4) if taus else np.nan,
        "spearman_rho": round(np.mean(rhos), 4) if rhos else np.nan,
        "top1_accuracy": round(top1_ok / n, 4) if n else np.nan,
        "top2_accuracy": round(top2_ok / n, 4) if n else np.nan,
        "n_scenarios_evaluated": n,
    }


def compute_failure_rate(arch_df):
    """Failure rate for Hybrid architecture."""
    if "extraction_failed" not in arch_df.columns:
        return {}

    n_total = arch_df["scenario_id"].nunique()
    n_ef = n_cf = n_any = 0

    for sid in arch_df["scenario_id"].unique():
        g = arch_df[arch_df["scenario_id"] == sid]
        ef = g["extraction_failed"].astype(str).str.lower().str.strip().eq("true").any()
        cf = ("gt_calculation_failed" in g.columns and
              g["gt_calculation_failed"].astype(str).str.lower().str.strip().eq("true").any())
        if ef: n_ef += 1
        if cf: n_cf += 1
        if ef or cf: n_any += 1

    return {
        "extraction_failure_rate": round(n_ef / n_total, 4) if n_total else 0,
        "calculation_failure_rate": round(n_cf / n_total, 4) if n_total else 0,
        "total_failure_rate": round(n_any / n_total, 4) if n_total else 0,
        "n_extraction_failures": n_ef,
        "n_calculation_failures": n_cf,
        "n_total_arch_scenarios": n_total,
    }

def evaluate_all(config):
    print("=" * 72)
    print("  MCDA ARCHITECTURE EVALUATION — METRICS REPORT")
    print("=" * 72)

    # 1. Load
    print("\n[1] Loading ground truth...")
    gt_by_type = load_ground_truth(config)
    for dt, df in gt_by_type.items():
        print(f"    {dt}: {df['scenario_id'].nunique()} scenarios, {len(df)} rows")

    print("\n[2] Loading architectures...")
    arch_dfs = {}
    for name, path in config["architectures"].items():
        arch_dfs[name] = load_architecture(path, name)
        dtc = arch_dfs[name]["decision_type"].value_counts().to_dict()
        print(f"    {name}: {arch_dfs[name]['scenario_id'].nunique()} scenarios {dtc}")

    # 2. Match
    print("\n[3] Matching...")
    gt_lookup = build_gt_lookup(gt_by_type)
    print(f"    GT lookup: {len(gt_lookup)} unique (question, location) keys")

    merged_dfs = {}
    for name, adf in arch_dfs.items():
        merged_dfs[name] = match_scenarios(gt_lookup, adf, name)

    print("  RESULTS")


    all_metrics = []

    for arch_name in ["Pure", "RAG", "Hybrid"]:
        merged = merged_dfs[arch_name]
        if len(merged) == 0:
            print(f"\n{arch_name}: No matched data")
            continue


        print(f"  {arch_name.upper()}")

        # Failure rate (Hybrid only; other two did not have previous failures)
        if arch_name == "Hybrid":
            fail = compute_failure_rate(arch_dfs[arch_name])
            if fail:
                print(f"\n  Failures: extraction={fail['n_extraction_failures']}"
                      f"/{fail['n_total_arch_scenarios']} "
                      f"({fail['extraction_failure_rate']*100:.1f}%), "
                      f"calc={fail['n_calculation_failures']}"
                      f"/{fail['n_total_arch_scenarios']} "
                      f"({fail['calculation_failure_rate']*100:.1f}%), "
                      f"total={fail['total_failure_rate']*100:.1f}%")
                for k, v in fail.items():
                    all_metrics.append({
                        "architecture": arch_name,
                        "decision_type": "Overall",
                        "metric": k, "value": v,
                    })

        crit = compute_criterion_metrics(merged)
        rank = compute_ranking_metrics(merged)
        n_eval = rank["n_scenarios_evaluated"]

        print(f"\n  OVERALL ({n_eval} scenarios):")
        print(f"    Criterion MAE / RMSE:")
        for c in CRITERIA:
            print(f"      {c:20s}  MAE={crit[f'{c}_MAE']:.4f}  "
                  f"RMSE={crit[f'{c}_RMSE']:.4f}")
        print(f"      {'OVERALL':20s}  MAE={crit['overall_MAE']:.4f}  "
              f"RMSE={crit['overall_RMSE']:.4f}")

        print(f"    Ranking:")
        print(f"      Kendall τ:  {rank['kendall_tau']:.4f}")
        print(f"      Spearman ρ: {rank['spearman_rho']:.4f}")
        print(f"      Top-1:      {rank['top1_accuracy']:.4f} "
              f"({int(rank['top1_accuracy'] * n_eval)}/{n_eval})")
        print(f"      Top-2:      {rank['top2_accuracy']:.4f} "
              f"({int(rank['top2_accuracy'] * n_eval)}/{n_eval})")

        # Store overall
        for k, v in {**crit, **rank}.items():
            all_metrics.append({
                "architecture": arch_name,
                "decision_type": "Overall",
                "metric": k, "value": v,
            })

        # per decision type
        for dtype in ["HVAC", "Appliance", "Shower"]:
            dt_data = merged[merged["decision_type"] == dtype]
            if len(dt_data) == 0:
                print(f"\n  {dtype}: No matched data")
                continue

            dt_crit = compute_criterion_metrics(dt_data)
            dt_rank = compute_ranking_metrics(dt_data)
            n_dt = dt_rank["n_scenarios_evaluated"]

            print(f"\n  {dtype} ({n_dt} scenarios, {len(dt_data)} alt rows):")
            print(f"    MAE:  EC={dt_crit['energy_cost_MAE']:.3f}  "
                  f"ENV={dt_crit['environmental_MAE']:.3f}  "
                  f"COM={dt_crit['comfort_MAE']:.3f}  "
                  f"PRA={dt_crit['practicality_MAE']:.3f}  "
                  f"All={dt_crit['overall_MAE']:.3f}")
            print(f"    RMSE: EC={dt_crit['energy_cost_RMSE']:.3f}  "
                  f"ENV={dt_crit['environmental_RMSE']:.3f}  "
                  f"COM={dt_crit['comfort_RMSE']:.3f}  "
                  f"PRA={dt_crit['practicality_RMSE']:.3f}  "
                  f"All={dt_crit['overall_RMSE']:.3f}")
            print(f"    τ={dt_rank['kendall_tau']:.4f}  "
                  f"ρ={dt_rank['spearman_rho']:.4f}  "
                  f"Top1={dt_rank['top1_accuracy']:.4f} "
                  f"({int(dt_rank['top1_accuracy']*n_dt)}/{n_dt})  "
                  f"Top2={dt_rank['top2_accuracy']:.4f} "
                  f"({int(dt_rank['top2_accuracy']*n_dt)}/{n_dt})")

            for k, v in {**dt_crit, **dt_rank}.items():
                all_metrics.append({
                    "architecture": arch_name,
                    "decision_type": dtype,
                    "metric": k, "value": v,
                })



    def _get(arch, dtype, metric):
        """Helper to pull a metric value from all_metrics list."""
        val = next(
            (m["value"] for m in all_metrics
             if m["architecture"] == arch
             and m["decision_type"] == dtype
             and m["metric"] == metric),
            np.nan
        )
        return val

    def _fmt(val, is_int=False):
        if isinstance(val, float) and np.isnan(val):
            return f"{'N/A':>10}"
        return f"{int(val):>10}" if is_int else f"{val:>10.4f}"

    archs = ["Pure", "RAG", "Hybrid"]

    # Overall table
    header = f"  {'Metric':<24}" + "".join(f"{a:>10}" for a in archs)
    print(f"\n{header}")
    print("  " + "-" * (24 + 10 * len(archs)))

    for metric in ["overall_MAE", "overall_RMSE", "kendall_tau", "spearman_rho",
                    "top1_accuracy", "top2_accuracy", "n_scenarios_evaluated"]:
        is_int = metric == "n_scenarios_evaluated"
        row = f"  {metric:<24}"
        for a in archs:
            row += _fmt(_get(a, "Overall", metric), is_int)
        print(row)

    # Per-criterion MAE
    print(f"\n  {'Criterion MAE':<24}" + "".join(f"{a:>10}" for a in archs))
    print("  " + "-" * (24 + 10 * len(archs)))
    for c in CRITERIA:
        row = f"  {c:<24}"
        for a in archs:
            row += _fmt(_get(a, "Overall", f"{c}_MAE"))
        print(row)

    # Kendall tau by decision type
    print(f"\n  {'Kendall τ by Type':<24}" + "".join(f"{a:>10}" for a in archs))
    print("  " + "-" * (24 + 10 * len(archs)))
    for dtype in ["HVAC", "Appliance", "Shower"]:
        row = f"  {dtype:<24}"
        for a in archs:
            row += _fmt(_get(a, dtype, "kendall_tau"))
        print(row)

    # Top-1 by decision type
    print(f"\n  {'Top-1 by Type':<24}" + "".join(f"{a:>10}" for a in archs))
    print("  " + "-" * (24 + 10 * len(archs)))
    for dtype in ["HVAC", "Appliance", "Shower"]:
        row = f"  {dtype:<24}"
        for a in archs:
            row += _fmt(_get(a, dtype, "top1_accuracy"))
        print(row)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(config["output_csv"], index=False)
    print(f"\n\nMetrics saved to: {config['output_csv']}")
    print(f"Total metric rows: {len(metrics_df)}")

    return metrics_df, merged_dfs

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python calculate_metrics.py")
        print("  Modify CONFIG dict at top of file to change paths.")
        sys.exit(0)

    metrics_df, merged_dfs = evaluate_all(CONFIG)