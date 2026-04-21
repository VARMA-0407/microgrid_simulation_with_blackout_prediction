"""
rl_evaluate.py — Evaluate a Trained RL Policy vs. Rule-Based Baseline.

Loads the trained PPO policy from outputs/rl_policy/best_model.zip,
runs a full simulation episode, and compares it against the rule-based
battery dispatch across:
  - LOLP (Loss of Load Probability)
  - EENS (Expected Energy Not Served)
  - Blackout count
  - SOC profile
  - Voltage profile (v_min)
  - Reward curve

Usage
-----
    python rl_evaluate.py
    python rl_evaluate.py --n_days 30 --model_path outputs/rl_policy/best_model
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stable_baselines3 import PPO

from rl_environment import MicrogridEnv
from digital_twin import run_physics_simulation


# ── Paths ──────────────────────────────────────────────────────────────────────
LOAD_PATH  = r'datasets\LD2011_2014.txt'
SOLAR_PATH = r'datasets\Plant_1_Generation_Data.csv'
POLICY_DIR = 'outputs/rl_policy'
PLOT_DIR   = 'outputs/rl_plots'


# ── Helper — run one deterministic episode with the RL policy ─────────────────

def run_rl_episode(model: PPO, n_days: int = 30) -> pd.DataFrame:
    """
    Run a full deterministic episode with the trained model.

    Returns
    -------
    pd.DataFrame with columns matching digital_twin simulation output.
    """
    env = MicrogridEnv(LOAD_PATH, SOLAR_PATH, n_days=n_days)
    obs, _ = env.reset()

    records = []
    done    = False
    t       = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        records.append({
            'timestep':     t,
            'hour_of_day':  t % 24,
            'day':          t // 24,
            'p_load_mw':    round(info['p_load_mw'],    6),
            'p_solar_mw':   round(info['p_solar_mw'],   6),
            'p_battery_mw': round(info['p_battery_mw'], 6),
            'soc':          round(info['soc'],           6),
            'v_min':        round(info['v_min'],         6),
            'blackout':     info['blackout'],
            'reward':       round(reward,                4),
        })
        t += 1

    env.close()
    return pd.DataFrame(records)


# ── Helper — reliability metrics ──────────────────────────────────────────────

def reliability_metrics(df: pd.DataFrame, label: str) -> dict:
    """Compute and print LOLP and EENS for a simulation DataFrame."""
    T              = len(df)
    blackout_hours = int(df['blackout'].sum())
    lolp           = blackout_hours / T if T > 0 else 0.0
    eens           = float(df.loc[df['blackout'] == 1, 'p_load_mw'].sum())

    print(f"\n  [{label}]")
    print(f"    Timesteps      : {T}")
    print(f"    Blackout hours : {blackout_hours}")
    print(f"    LOLP           : {lolp * 100:.2f}%")
    print(f"    EENS           : {eens:.3f} MWh")

    return {'label': label, 'T': T, 'blackouts': blackout_hours,
            'lolp': lolp, 'eens': eens}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(df_rule: pd.DataFrame, df_rl: pd.DataFrame):
    """
    Save a 4-panel comparison figure:
      Panel 1 — Blackout events timeline
      Panel 2 — SOC profile
      Panel 3 — Voltage profile (v_min)
      Panel 4 — Cumulative reward (RL only)
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('RL Agent vs Rule-Based Dispatch — Microgrid Comparison',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    t_rule = df_rule['timestep'].values
    t_rl   = df_rl['timestep'].values
    COLORS = {'rule': '#E74C3C', 'rl': '#2ECC71'}

    # ── Panel 1: Blackout events ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(t_rule, df_rule['blackout'].values, alpha=0.55,
                     color=COLORS['rule'], label='Rule-Based')
    ax1.fill_between(t_rl,   df_rl['blackout'].values,   alpha=0.55,
                     color=COLORS['rl'],   label='RL Agent')
    ax1.set_ylabel('Blackout (1=yes)')
    ax1.set_title('Blackout Events')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, max(t_rule[-1], t_rl[-1]))
    ax1.set_yticks([0, 1])
    ax1.grid(axis='x', alpha=0.3)

    # ── Panel 2: SOC profile ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_rule, df_rule['soc'].values, color=COLORS['rule'],
             lw=1.2, alpha=0.85, label='Rule-Based SOC')
    ax2.plot(t_rl,   df_rl['soc'].values,   color=COLORS['rl'],
             lw=1.2, alpha=0.85, label='RL Agent SOC')
    ax2.axhline(0.2, color='grey', ls='--', lw=0.8, label='SOC min (0.2)')
    ax2.axhline(0.8, color='grey', ls=':',  lw=0.8, label='SOC max (0.8)')
    ax2.set_ylabel('SOC (fraction)')
    ax2.set_title('Battery State of Charge')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, max(t_rule[-1], t_rl[-1]))
    ax2.set_ylim(0, 1)
    ax2.grid(axis='x', alpha=0.3)

    # ── Panel 3: Voltage profile ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t_rule, df_rule['v_min'].values, color=COLORS['rule'],
             lw=1.0, alpha=0.85, label='Rule-Based v_min')
    ax3.plot(t_rl,   df_rl['v_min'].values,   color=COLORS['rl'],
             lw=1.0, alpha=0.85, label='RL Agent v_min')
    ax3.axhline(0.92, color='red', ls='--', lw=0.9, label='V limit (0.92 pu)')
    ax3.set_ylabel('Min Bus Voltage (pu)')
    ax3.set_title('Minimum Bus Voltage')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_xlim(0, max(t_rule[-1], t_rl[-1]))
    ax3.grid(axis='x', alpha=0.3)

    # ── Panel 4: Cumulative reward (RL only) ─────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    cum_reward = np.cumsum(df_rl['reward'].values)
    ax4.plot(t_rl, cum_reward, color='#3498DB', lw=1.4, label='RL cumulative reward')
    ax4.axhline(0, color='grey', lw=0.7, ls='--')
    ax4.set_xlabel('Timestep (hours)')
    ax4.set_ylabel('Cumulative Reward')
    ax4.set_title('RL Agent Cumulative Reward')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_xlim(0, t_rl[-1])
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(PLOT_DIR, 'rl_vs_rule_comparison.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison plot saved: {out_path}")


def plot_summary_bar(metrics_list: list):
    """Bar chart comparing LOLP and EENS for rule-based vs. RL."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    labels    = [m['label']    for m in metrics_list]
    lolp_vals = [m['lolp']*100 for m in metrics_list]
    eens_vals = [m['eens']     for m in metrics_list]
    colors    = ['#E74C3C', '#2ECC71']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Reliability Metrics: RL Agent vs Rule-Based', fontsize=13, fontweight='bold')

    bars1 = ax1.bar(labels, lolp_vals, color=colors, edgecolor='white', width=0.5)
    ax1.set_ylabel('LOLP (%)')
    ax1.set_title('Loss of Load Probability')
    for bar, val in zip(bars1, lolp_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(lolp_vals) * 1.3 + 1)

    bars2 = ax2.bar(labels, eens_vals, color=colors, edgecolor='white', width=0.5)
    ax2.set_ylabel('EENS (MWh)')
    ax2.set_title('Expected Energy Not Served')
    for bar, val in zip(bars2, eens_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max(eens_vals) * 1.3 + 0.01)

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, 'reliability_summary.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Summary bar chart saved: {out_path}")


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate(model_path: str = None, n_days: int = 30):
    """
    Load the trained PPO model, run a full episode, compare vs. rule-based.

    Parameters
    ----------
    model_path : str | None
        Path to the saved model (without .zip). Defaults to best_model.
    n_days : int
        Simulation horizon (days) for both methods.
    """
    if model_path is None:
        model_path = os.path.join(POLICY_DIR, 'best_model')

    if not os.path.exists(model_path + '.zip'):
        print(f"[ERROR] Trained model not found: {model_path}.zip")
        print("        Run  python rl_train.py  first.")
        return

    print("=" * 60)
    print("  RL EVALUATION & COMPARISON")
    print("=" * 60)
    print(f"  Model      : {model_path}.zip")
    print(f"  Days       : {n_days}")

    # ── Load trained model ────────────────────────────────────────────────
    print("\n[1/4] Loading trained PPO model...")
    model = PPO.load(model_path)
    print("      Done.")

    # ── Run RL episode ────────────────────────────────────────────────────
    print("\n[2/4] Running RL agent episode...")
    df_rl = run_rl_episode(model, n_days=n_days)
    print(f"      {len(df_rl)} timesteps collected.")
    df_rl.to_csv('outputs/rl_simulation_results.csv', index=False)
    print("      Saved: outputs/rl_simulation_results.csv")

    # ── Run rule-based baseline ───────────────────────────────────────────
    print("\n[3/4] Running rule-based baseline (digital_twin.py)...")
    df_rule = run_physics_simulation(LOAD_PATH, SOLAR_PATH, n_days=n_days)
    # Ensure reward column exists for compatibility (rule-based has no reward)
    if 'reward' not in df_rule.columns:
        df_rule['reward'] = 0.0

    # ── Compare metrics ───────────────────────────────────────────────────
    print("\n[4/4] Reliability comparison:")
    m_rule = reliability_metrics(df_rule, 'Rule-Based')
    m_rl   = reliability_metrics(df_rl,   'RL Agent')

    lolp_imp = (m_rule['lolp'] - m_rl['lolp']) / (m_rule['lolp'] + 1e-9) * 100
    eens_imp = (m_rule['eens'] - m_rl['eens']) / (m_rule['eens'] + 1e-9) * 100

    print(f"\n  LOLP improvement : {lolp_imp:+.1f}%")
    print(f"  EENS improvement : {eens_imp:+.1f}%")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating comparison plots...")
    plot_comparison(df_rule, df_rl)
    plot_summary_bar([m_rule, m_rl])

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  RL results CSV  : outputs/rl_simulation_results.csv")
    print(f"  Plots directory : {PLOT_DIR}/")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent vs rule-based baseline')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (without .zip). Defaults to best_model.')
    parser.add_argument('--n_days', type=int, default=30,
                        help='Simulation horizon in days (default: 30)')
    args = parser.parse_args()

    evaluate(model_path=args.model_path, n_days=args.n_days)