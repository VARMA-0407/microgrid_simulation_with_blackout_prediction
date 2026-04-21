"""
main.py — Full Pipeline Entry Point for the Microgrid Digital Twin.

Phases
------
  1. Rule-based Physics Simulation   → simulation_results.csv
  2. ML Model Training & Evaluation  → outputs/model/best_model.pkl
  3. Reliability Metrics             → LOLP, EENS (rule-based baseline)
  4. RL Agent Training               → outputs/rl_policy/best_model.zip
  5. RL-Optimised Simulation         → outputs/rl_simulation_results.csv
  6. RL vs Baseline Evaluation       → outputs/rl_plots/

Usage
-----
    python main.py                 # full pipeline (rule-based + RL)
    python main.py --skip-rl       # only rule-based + ML (faster)
    python main.py --only-rl       # assume rule-based done; only run RL phases
    python main.py --rl-timesteps 50000   # quick RL smoke test (50k steps)
"""

import os
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
LOAD_PATH  = r'datasets\LD2011_2014.txt'
SOLAR_PATH = r'datasets\Plant_1_Generation_Data.csv'
N_DAYS     = 30


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Phases
# ─────────────────────────────────────────────────────────────────────────────

def phase_rule_based():
    """Phase 1–3: Rule-based simulation + ML training + reliability metrics."""
    from digital_twin import run_digital_twin
    from ml_model import compute_reliability_metrics

    print("\n" + "=" * 60)
    print("  PHASE 1 — Rule-Based Physics Simulation + ML Training")
    print("=" * 60)
    df = run_digital_twin(LOAD_PATH, SOLAR_PATH, n_days=N_DAYS, use_rl=False)

    print("\n" + "=" * 60)
    print("  PHASE 2 — Reliability Metrics (Rule-Based Baseline)")
    print("=" * 60)
    metrics_rule = compute_reliability_metrics(df)

    return df, metrics_rule


def phase_rl_train(timesteps: int = 300_000, n_envs: int = 2):
    """Phase 4: Train the PPO RL agent inside MicrogridEnv."""
    try:
        from rl_train import train
    except ImportError as e:
        print(f"\n  [ERROR] Cannot import RL modules: {e}")
        print(f"          Install with: pip install gymnasium stable-baselines3")
        print(f"          Make sure you're using the correct Python/venv.")
        return

    print("\n" + "=" * 60)
    print("  PHASE 4 — RL Agent Training (PPO)")
    print("=" * 60)
    print(f"  Timesteps : {timesteps:,}")
    train(timesteps=timesteps, n_days=N_DAYS, n_envs=n_envs)


def phase_rl_simulate():
    """Phase 5: Re-run the simulation with the trained RL policy."""
    from digital_twin import run_digital_twin

    print("\n" + "=" * 60)
    print("  PHASE 5 — RL-Policy Simulation")
    print("=" * 60)
    df_rl = run_digital_twin(
        LOAD_PATH, SOLAR_PATH,
        n_days        = N_DAYS,
        use_rl        = True,
        rl_policy_path= 'outputs/rl_policy/best_model',
    )
    return df_rl


def phase_rl_evaluate(metrics_rule=None):
    """Phase 6: Compare RL policy vs rule-based baseline with plots."""
    try:
        from rl_evaluate import evaluate
    except ImportError as e:
        print(f"\n  [ERROR] Cannot import RL evaluation: {e}")
        return

    print("\n" + "=" * 60)
    print("  PHASE 6 — RL Evaluation & Comparison")
    print("=" * 60)
    evaluate(n_days=N_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Microgrid Digital Twin — full pipeline'
    )
    parser.add_argument(
        '--skip-rl', action='store_true',
        help='Skip RL phases; run only rule-based simulation + ML.'
    )
    parser.add_argument(
        '--only-rl', action='store_true',
        help='Skip rule-based phases; run only RL training + evaluation.'
    )
    parser.add_argument(
        '--rl-timesteps', type=int, default=300_000,
        help='PPO training timesteps (default: 300000; use 50000 for a quick test).'
    )
    parser.add_argument(
        '--n-envs', type=int, default=2,
        help='Number of parallel RL training environments (default: 2).'
    )
    args = parser.parse_args()

    os.makedirs('outputs',       exist_ok=True)
    os.makedirs('outputs/model', exist_ok=True)

    metrics_rule = None

    # ── Rule-based pipeline ───────────────────────────────────────────────
    if not args.only_rl:
        _, metrics_rule = phase_rule_based()

    # ── RL pipeline ───────────────────────────────────────────────────────
    if not args.skip_rl:
        phase_rl_train(timesteps=args.rl_timesteps, n_envs=args.n_envs)
        phase_rl_simulate()
        phase_rl_evaluate(metrics_rule)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FULL PIPELINE COMPLETE")
    print("=" * 60)
    print("  Outputs:")
    print("    outputs/simulation_results.csv      <- rule-based")
    print("    outputs/rl_simulation_results.csv   <- RL policy")
    print("    outputs/model/best_model.pkl        <- ML classifier")
    print("    outputs/rl_policy/best_model.zip    <- trained PPO policy")
    print("    outputs/rl_plots/                   <- comparison charts")
    if not args.skip_rl:
        print("\n  To view training curves:")
        print("    tensorboard --logdir outputs/rl_logs/")
    print("=" * 60)


if __name__ == '__main__':
    main()
