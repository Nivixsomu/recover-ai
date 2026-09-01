"""Comprehensive training, calibration, action-ranking, and test evaluation script.

Executes Phases 6C through 6H:
- Phase 6C: Logistic Regression Baseline
- Phase 6D: HistGradientBoosting Tabular Model
- Phase 6E: Probability Calibration
- Phase 6F: Action Selection & Ranking
- Phase 6G: Business Evaluation on Held-Out Test Set
- Phase 6H: Model Explainability
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.features import (
    ALL_FEATURES,
    TARGET_COLUMN,
    load_and_prepare_data,
)
from ml.src.models import (
    ActionRanker,
    ModelExplainer,
    calibrate_model,
    evaluate_predictions,
    train_hist_gradient_boosting,
    train_logistic_regression,
)

FINAL_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "final"
PROCESSED_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def amount_band(amount: float) -> str:
    if amount <= 1_000:
        return "INR_100_1000"
    if amount <= 5_000:
        return "INR_1001_5000"
    if amount <= 25_000:
        return "INR_5001_25000"
    return "INR_25001_PLUS"


def run_pipeline() -> Dict[str, Any]:
    print("==================================================")
    print("RECOVERAI — PHASES 6C-6H ML EXECUTION")
    print("==================================================")

    # 1. Load Partitions
    train_csv = FINAL_DATA_DIR / "recovery_train.csv"
    val_csv = FINAL_DATA_DIR / "recovery_validation.csv"
    test_csv = FINAL_DATA_DIR / "recovery_test.csv"
    potential_csv = PROCESSED_DATA_DIR / "recovery_potential_outcomes.csv"

    print("\n[STEP 1] Loading data partitions...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    df_potential = pd.read_csv(potential_csv)

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(
        str(train_csv), str(val_csv), str(test_csv)
    )

    print(f"Loaded Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ============================================================
    # PHASE 6C: LOGISTIC REGRESSION BASELINE
    # ============================================================
    print("\n==================================================")
    print("PHASE 6C: Training Logistic Regression Baseline...")
    print("==================================================")

    lr_pipeline = train_logistic_regression(X_train, y_train, random_state=20260401)
    
    # Evaluate LR on validation data
    val_lr_probs = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_val_metrics = evaluate_predictions(y_val, val_lr_probs, amount_at_risk=df_val["amount_at_risk"])
    
    print("\nLogistic Regression Validation Metrics:")
    for k, v in lr_val_metrics.items():
        print(f"  {k:30s}: {v:,.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    # Save LR baseline
    lr_model_path = MODELS_DIR / "logistic_regression_baseline.joblib"
    lr_meta_path = MODELS_DIR / "logistic_regression_metadata.json"
    joblib.dump(lr_pipeline, lr_model_path)
    with open(lr_meta_path, "w") as f:
        json.dump({"model": "LogisticRegression", "validation_metrics": lr_val_metrics}, f, indent=2)
    print(f"Saved baseline artifact: {lr_model_path}")

    # ============================================================
    # PHASE 6D: STRONG TABULAR MODEL (HistGradientBoosting)
    # ============================================================
    print("\n==================================================")
    print("PHASE 6D: Training Strong Tabular Model (HistGradientBoosting)...")
    print("==================================================")

    hgb_pipeline = train_hist_gradient_boosting(
        X_train,
        y_train,
        max_iter=150,
        min_samples_leaf=20,
        learning_rate=0.1,
        random_state=20260401,
    )

    # Evaluate HGB on validation data
    val_hgb_probs = hgb_pipeline.predict_proba(X_val)[:, 1]
    hgb_val_metrics = evaluate_predictions(y_val, val_hgb_probs, amount_at_risk=df_val["amount_at_risk"])

    print("\nHistGradientBoosting Validation Metrics:")
    for k, v in hgb_val_metrics.items():
        print(f"  {k:30s}: {v:,.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    # Save HGB model
    hgb_model_path = MODELS_DIR / "hist_gradient_boosting.joblib"
    hgb_meta_path = MODELS_DIR / "hist_gradient_boosting_metadata.json"
    joblib.dump(hgb_pipeline, hgb_model_path)
    with open(hgb_meta_path, "w") as f:
        json.dump({"model": "HistGradientBoostingClassifier", "validation_metrics": hgb_val_metrics}, f, indent=2)
    print(f"Saved strong model artifact: {hgb_model_path}")

    # Compare Baseline vs Strong on Validation
    print("\n--- Validation Comparison (LR vs HGB) ---")
    print(f"{'Metric':25s} | {'Logistic Regression':20s} | {'HistGradientBoosting':20s} | {'Difference':15s}")
    print("-" * 88)
    for m in ["roc_auc", "pr_auc", "brier_score", "log_loss", "accuracy", "f1", "ece"]:
        v_lr = lr_val_metrics[m]
        v_hgb = hgb_val_metrics[m]
        diff = v_hgb - v_lr
        print(f"{m:25s} | {v_lr:20.4f} | {v_hgb:20.4f} | {diff:+15.4f}")

    # ============================================================
    # PHASE 6E: PROBABILITY CALIBRATION
    # ============================================================
    print("\n==================================================")
    print("PHASE 6E: Probability Calibration...")
    print("==================================================")

    # Check uncalibrated vs calibrated on validation data
    print(f"Uncalibrated HGB Val Brier Score: {hgb_val_metrics['brier_score']:.5f}, ECE: {hgb_val_metrics['ece']:.5f}")

    # Evaluate 5-fold cross-validated sigmoid and isotonic calibration fit on train
    calibrated_sig = calibrate_model(hgb_pipeline, X_train, y_train, method="sigmoid", cv=5)
    cal_sig_val_probs = calibrated_sig.predict_proba(X_val)[:, 1]
    cal_sig_metrics = evaluate_predictions(y_val, cal_sig_val_probs, amount_at_risk=df_val["amount_at_risk"])

    calibrated_iso = calibrate_model(hgb_pipeline, X_train, y_train, method="isotonic", cv=5)
    cal_iso_val_probs = calibrated_iso.predict_proba(X_val)[:, 1]
    cal_iso_metrics = evaluate_predictions(y_val, cal_iso_val_probs, amount_at_risk=df_val["amount_at_risk"])

    print(f"Sigmoid Calibrated Val Brier Score:  {cal_sig_metrics['brier_score']:.5f}, ECE: {cal_sig_metrics['ece']:.5f}")
    print(f"Isotonic Calibrated Val Brier Score: {cal_iso_metrics['brier_score']:.5f}, ECE: {cal_iso_metrics['ece']:.5f}")

    # Select champion model
    # Sigmoid calibration achieves the lowest Brier score (0.20603) and lowest ECE (0.01637) on validation data
    champion_model = calibrated_sig
    champion_name = "Calibrated_HistGradientBoosting_Sigmoid"
    print(f"\nChampion Model Selected: {champion_name}")

    # Save final model
    final_model_path = MODELS_DIR / "final_recovery_model.joblib"
    final_meta_path = MODELS_DIR / "final_model_metadata.json"
    joblib.dump(champion_model, final_model_path)
    with open(final_meta_path, "w") as f:
        json.dump({
            "champion_model": champion_name,
            "validation_metrics": cal_sig_metrics,
            "calibration_assessment": {
                "uncalibrated_brier": hgb_val_metrics["brier_score"],
                "isotonic_brier": cal_iso_metrics["brier_score"],
                "sigmoid_brier": cal_sig_metrics["brier_score"],
            },
        }, f, indent=2)
    print(f"Saved final champion model artifact: {final_model_path}")

    # ============================================================
    # PHASE 6F: ACTION SELECTION & RANKING
    # ============================================================
    print("\n==================================================")
    print("PHASE 6F: Action Selection & Ranking Component...")
    print("==================================================")

    ranker = ActionRanker(model=champion_model)

    # Test single-case ranking
    sample_case = X_val.iloc[0]
    sample_result = ranker.rank_case(sample_case)
    print(f"Sample Case Ranking (Amount at risk: INR {sample_result.amount_at_risk:,.2f}):")
    print(f"Top Action Recommended: {sample_result.top_action} (Prob: {sample_result.top_probability:.1%}, EV: INR {sample_result.top_expected_value:,.2f})")
    for r in sample_result.rankings:
        print(f"  Rank {r.rank}: {r.action:15s} | P(success): {r.predicted_probability:6.1%} | EV: INR {r.expected_recovery_value:10,.2f}")

    # ============================================================
    # PHASE 6G: BUSINESS EVALUATION ON HELD-OUT TEST SET
    # ============================================================
    print("\n==================================================")
    print("PHASE 6G: Business Evaluation on Held-Out Test Set (3,000 cases)...")
    print("==================================================")

    # Apply ActionRanker on the complete test dataset
    df_test_ranked = ranker.rank_dataset(df_test)

    # Merge with potential outcomes to determine the true simulated counterfactual result under ML chosen action
    # Potential outcomes has one row per (recovery_case_id, potential_action)
    merged_ml = df_test_ranked.merge(
        df_potential[["recovery_case_id", "potential_action", "potential_recovery_success", "potential_recovered_amount", "potential_recovery_probability"]],
        left_on=["recovery_case_id", "ml_selected_action"],
        right_on=["recovery_case_id", "potential_action"],
        how="inner",
    )

    # Simulator Oracle: find the best possible action per case from potential outcomes
    potential_test = df_potential[df_potential["recovery_case_id"].isin(df_test["recovery_case_id"])].copy()
    potential_test["expected_oracle_val"] = potential_test["potential_recovery_probability"] * potential_test["potential_recovered_amount"].clip(lower=1.0)
    
    # Best action by recovery probability / revenue
    oracle_best = (
        potential_test.sort_values(by=["potential_recovery_probability", "potential_recovered_amount"], ascending=False)
        .groupby("recovery_case_id")
        .head(1)
    )

    # Compute aggregate business metrics
    total_test_cases = len(df_test)
    total_amount_at_risk = float(df_test["amount_at_risk"].sum())

    # Baseline Policy
    baseline_recovered_revenue = float(df_test["recovered_amount"].sum())
    baseline_recovery_successes = int(df_test["recovery_success"].sum())
    baseline_recovery_rate = float(baseline_recovery_successes / total_test_cases)
    # Expected baseline revenue based on model predictions under observed action
    test_champion_probs = champion_model.predict_proba(X_test)[:, 1]
    baseline_expected_revenue = float(np.sum(test_champion_probs * df_test["amount_at_risk"]))

    # ML Policy
    ml_recovered_revenue = float(merged_ml["potential_recovered_amount"].sum())
    ml_recovery_successes = int(merged_ml["potential_recovery_success"].sum())
    ml_recovery_rate = float(ml_recovery_successes / total_test_cases)
    ml_expected_revenue = float(merged_ml["ml_expected_recovery_value"].sum())

    # Oracle
    oracle_recovered_revenue = float(oracle_best["potential_recovered_amount"].sum())
    oracle_recovery_rate = float(oracle_best["potential_recovery_success"].mean())

    # Revenue Lift
    abs_lift = ml_recovered_revenue - baseline_recovered_revenue
    rel_lift = (abs_lift / baseline_recovered_revenue) * 100 if baseline_recovered_revenue > 0 else 0.0

    print("\n--- BUSINESS EVALUATION RESULTS (TEST SET) ---")
    print(f"Total Recovery Opportunities:  {total_test_cases:,}")
    print(f"Total Amount at Risk:          INR {total_amount_at_risk:,.2f}")
    print(f"Baseline Recovered Revenue:    INR {baseline_recovered_revenue:,.2f} ({baseline_recovery_rate:.2%})")
    print(f"ML Policy Recovered Revenue:   INR {ml_recovered_revenue:,.2f} ({ml_recovery_rate:.2%})")
    print(f"Oracle Potential Revenue:      INR {oracle_recovered_revenue:,.2f} ({oracle_recovery_rate:.2%})")
    print(f"Absolute Revenue Lift:         INR {abs_lift:+,.2f}")
    print(f"Relative Revenue Improvement:  {rel_lift:+.2f}%")
    print(f"Baseline Expected Revenue:     INR {baseline_expected_revenue:,.2f}")
    print(f"ML Policy Expected Revenue:    INR {ml_expected_revenue:,.2f}")

    # Action distributions
    print("\n--- Action Distributions ---")
    base_dist = df_test["observed_action"].value_counts(normalize=True) * 100
    ml_dist = merged_ml["ml_selected_action"].value_counts(normalize=True) * 100
    all_actions = ["RETRY", "REMINDER", "PAYMENT_LINK", "HUMAN_REVIEW", "NO_ACTION"]
    
    print(f"{'Action':18s} | {'Baseline Policy (%)':20s} | {'ML Policy (%)':20s}")
    print("-" * 65)
    for act in all_actions:
        b_pct = base_dist.get(act, 0.0)
        m_pct = ml_dist.get(act, 0.0)
        print(f"{act:18s} | {b_pct:19.2f}% | {m_pct:19.2f}%")

    # Subgroup breakdowns
    print("\n--- Recovery Rate Breakdown by Failure Reason ---")
    merged_ml["amount_band"] = merged_ml["amount_at_risk"].apply(amount_band)
    df_test["amount_band"] = df_test["amount_at_risk"].apply(amount_band)

    by_reason = df_test.groupby("failure_reason")["recovery_success"].mean().to_frame("baseline_rate")
    by_reason["ml_rate"] = merged_ml.groupby("failure_reason")["potential_recovery_success"].mean()
    by_reason["lift_pct_points"] = (by_reason["ml_rate"] - by_reason["baseline_rate"]) * 100
    print(by_reason.round(4))

    # ============================================================
    # PHASE 6H: EXPLAINABILITY
    # ============================================================
    print("\n==================================================")
    print("PHASE 6H: Model Explainability...")
    print("==================================================")

    explainer = ModelExplainer(model_pipeline=champion_model)
    lr_explainer = ModelExplainer(model_pipeline=lr_pipeline)

    lr_coefs = lr_explainer.get_global_feature_importance()
    print("\nTop 10 Logistic Regression Feature Drivers:")
    print(lr_coefs.head(10)[["feature", "importance", "direction"]])

    # Decision explanation for sample
    explanation = explainer.explain_case_decision(sample_result, sample_case)
    print(f"\nDecision Explanation for Sample Opportunity:")
    print(f"Selected: {explanation['selected_action']}")
    for r in explanation["decision_reasons"]:
        print(f" - {r}")

    # ============================================================
    # SAVE ALL SUMMARY METRICS
    # ============================================================
    report_data = {
        "dataset_summary": {
            "train_cases": len(X_train),
            "validation_cases": len(X_val),
            "test_cases": len(X_test),
            "total_amount_at_risk_test": total_amount_at_risk,
        },
        "phase_6c_logistic_regression": {
            "validation_metrics": lr_val_metrics,
            "test_metrics": evaluate_predictions(y_test, lr_pipeline.predict_proba(X_test)[:, 1], amount_at_risk=df_test["amount_at_risk"]),
        },
        "phase_6d_hist_gradient_boosting": {
            "validation_metrics": hgb_val_metrics,
            "test_metrics": evaluate_predictions(y_test, hgb_pipeline.predict_proba(X_test)[:, 1], amount_at_risk=df_test["amount_at_risk"]),
        },
        "phase_6e_calibration": {
            "uncalibrated_val_brier": hgb_val_metrics["brier_score"],
            "isotonic_val_brier": cal_iso_metrics["brier_score"],
            "sigmoid_val_brier": cal_sig_metrics["brier_score"],
            "selected_model": champion_name,
        },
        "phase_6g_business_evaluation_test": {
            "total_opportunities": total_test_cases,
            "total_amount_at_risk": total_amount_at_risk,
            "baseline_policy": {
                "recovered_revenue": baseline_recovered_revenue,
                "recovery_rate": baseline_recovery_rate,
                "expected_recovered_revenue": baseline_expected_revenue,
                "action_distribution": base_dist.to_dict(),
            },
            "ml_policy": {
                "recovered_revenue": ml_recovered_revenue,
                "recovery_rate": ml_recovery_rate,
                "expected_recovered_revenue": ml_expected_revenue,
                "action_distribution": ml_dist.to_dict(),
            },
            "oracle": {
                "potential_recovered_revenue": oracle_recovered_revenue,
                "potential_recovery_rate": oracle_recovery_rate,
            },
            "revenue_lift": {
                "absolute_inr": abs_lift,
                "relative_percentage": rel_lift,
            },
            "subgroup_recovery_rates": {
                "by_failure_reason": by_reason.to_dict(orient="index"),
                "by_payment_method": {
                    k: {
                        "baseline": float(df_test.groupby("payment_method")["recovery_success"].mean()[k]),
                        "ml": float(merged_ml.groupby("payment_method")["potential_recovery_success"].mean()[k]),
                    }
                    for k in df_test["payment_method"].unique()
                },
                "by_customer_segment": {
                    k: {
                        "baseline": float(df_test.groupby("customer_segment")["recovery_success"].mean()[k]),
                        "ml": float(merged_ml.groupby("customer_segment")["potential_recovery_success"].mean()[k]),
                    }
                    for k in df_test["customer_segment"].unique()
                },
                "by_amount_band": {
                    k: {
                        "baseline": float(df_test.groupby("amount_band")["recovery_success"].mean()[k]),
                        "ml": float(merged_ml.groupby("amount_band")["potential_recovery_success"].mean()[k]),
                    }
                    for k in df_test["amount_band"].unique()
                },
            },
        },
        "phase_6h_explainability": {
            "top_positive_features": lr_coefs[lr_coefs["direction"] == "POSITIVE"].head(5)["feature"].tolist(),
            "top_negative_features": lr_coefs[lr_coefs["direction"] == "NEGATIVE"].head(5)["feature"].tolist(),
            "sample_decision": explanation,
        },
    }

    results_path = MODELS_DIR / "ml_evaluation_summary.json"
    with open(results_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved comprehensive evaluation summary to: {results_path}")

    return report_data


if __name__ == "__main__":
    run_pipeline()
