"""
Digital Twin Microgrid

Orchestrates the full microgrid simulation:
  1. Load p_load(t) and p_solar(t) from data modules
  2. Initialize battery and IEEE 33-bus network
  3. For each timestep t:
     - Battery dispatch  (rule-based OR trained RL policy)
     - AC load flow via pandapower
     - Blackout label generation
  4. Post-process: add rolling/lag temporal features
  5. Train ML predictor on first 80%, apply across full timeline
  6. Save results to CSV

RL mode (use_rl=True)
---------------------
When a trained PPO policy exists at `rl_policy_path`, run_digital_twin()
replaces the rule-based battery.step() with:
    obs            = env._get_obs()
    action, _      = rl_policy.predict(obs, deterministic=True)
    p_battery, soc = env._apply_action(float(action[0]))
exactly as specified in RL_INTEGRATION_GUIDE.md Section 12.

The output DataFrame is identical in both modes so add_temporal_features()
and ml_model.py work unchanged on RL-optimised simulation data.
"""

import numpy as np
import pandas as pd
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_load import prepare_load_series
from data_solar import prepare_solar_series
from battery_model import BatteryModel
from grid_simulator import create_ieee33_network, run_load_flow
from label_generator import generate_blackout_label, generate_severity_label

# Optional RL imports — only needed when use_rl=True
try:
    from stable_baselines3 import PPO
    from rl_environment import MicrogridEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


# ── Temporal feature engineering ─────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling/lag features. Called on the FULL dataset after simulation,
    and also used during live prediction (rolling window on running history).
    """
    df = df.copy()
    df['p_load_lag_1h']        = df['p_load_mw'].shift(1)
    df['p_load_lag_3h']        = df['p_load_mw'].shift(3)
    df['p_solar_lag_1h']       = df['p_solar_mw'].shift(1)
    df['soc_lag_1h']           = df['soc'].shift(1)
    df['load_rolling_avg_6h']  = df['p_load_mw'].rolling(6, min_periods=1).mean()
    df['solar_rolling_avg_6h'] = df['p_solar_mw'].rolling(6, min_periods=1).mean()
    df['v_min_rolling_min_6h'] = df['v_min'].rolling(6, min_periods=1).min()
    df['soc_rate_of_change']   = df['soc'].diff(1)
    df['load_rate_of_change']  = df['p_load_mw'].diff(1)
    df['load_solar_diff']      = df['p_load_mw'] - df['p_solar_mw']
    ml = df['p_load_mw'].max() or 1.0
    ms = df['p_solar_mw'].max() or 1.0
    df['grid_stress_index'] = (
        df['p_load_mw'] / ml
        + (1.0 - df['p_solar_mw'] / ms)
        + (1.0 - df['soc'])
    ) / 3.0
    # Engineered
    df['solar_load_ratio']   = df['p_solar_mw'] / (df['p_load_mw'] + 1e-6)
    df['soc_deficit']        = 1.0 - df['soc']
    df['v_violation_margin'] = df['v_min'] - 0.92
    df['net_power_balance']  = df['p_solar_mw'] + df['p_battery_mw'] - df['p_load_mw']
    df['is_night']           = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 20)).astype(int)
    lag_cols = ['p_load_lag_1h','p_load_lag_3h','p_solar_lag_1h','soc_lag_1h',
                'soc_rate_of_change','load_rate_of_change']
    df[lag_cols] = df[lag_cols].bfill().ffill()
    return df


# Features used by the ML predictor.
# IMPORTANT — What is excluded and why:
#   v_min, v_max, v_mean  — direct label components (blackout <- v_min < 0.92)
#   p_grid_mw             — direct label component  (blackout <- p_grid > 3.5)
#   converged             — direct label component  (blackout <- not converged)
#   soc                   — battery state is controlled deterministically;
#                           including it collapses to near-perfect lookup.
# Using only observable inputs that an operator would see BEFORE the
# load-flow runs, so the model is truly forecasting, not explaining.
FEATURE_COLS = [
    'p_load_mw',
    'p_solar_mw',
    'hour_of_day',
    'is_night',

    'p_load_lag_1h',
    'p_load_lag_3h',
    'p_solar_lag_1h',
    'soc_lag_1h',           # lagged SOC (known, not current)

    'load_rolling_avg_6h',
    'solar_rolling_avg_6h',

    'soc_rate_of_change',
    'load_rate_of_change',
    'load_solar_diff',
    'grid_stress_index',
]

# How many steps ahead the ML model predicts.
# 1 = predict next hour's blackout from current state (true forecasting).
PREDICT_AHEAD = 1


# ── RL simulation helper ─────────────────────────────────────────────────────

def _run_rl_simulation(
    load_path: str,
    solar_path: str,
    n_days: int,
    battery_capacity: float,
    rl_policy_path: str,
) -> pd.DataFrame:
    """
    Run simulation driven by a trained RL policy.

    Implements the integration pattern from RL_INTEGRATION_GUIDE.md §12:

        obs            = env._get_obs()
        action, _      = rl_policy.predict(obs, deterministic=True)
        p_battery, soc = env._apply_action(float(action[0]))

    Returns a DataFrame with the same columns as the rule-based path so
    add_temporal_features() and ml_model.py work unchanged.
    """
    if not RL_AVAILABLE:
        raise ImportError(
            "stable-baselines3 or gymnasium not installed. "
            "Run: pip install gymnasium stable-baselines3"
        )

    policy_zip = rl_policy_path if rl_policy_path.endswith('.zip') else rl_policy_path + '.zip'
    if not os.path.exists(policy_zip):
        raise FileNotFoundError(
            f"Trained RL policy not found: {policy_zip}\n"
            f"Run  python rl_train.py  first to produce the model."
        )

    print(f"[RL] Loading trained PPO policy from: {policy_zip}")
    rl_policy = PPO.load(rl_policy_path)

    print(f"[RL] Creating MicrogridEnv (n_days={n_days}) ...")
    env = MicrogridEnv(
        load_path        = load_path,
        solar_path       = solar_path,
        n_days           = n_days,
        battery_capacity = battery_capacity,
    )
    obs, _ = env.reset()
    T = env.T
    print(f"[RL] Episode length: {T} timesteps")

    records = []

    for t in range(T):
        # ── Guide §12: obs -> policy -> apply_action ────────────────────
        obs            = env._get_obs()
        action, _      = rl_policy.predict(obs, deterministic=True)
        p_cmd          = float(action[0]) * env.battery.charge_rate
        p_battery, soc = env._apply_action(p_cmd)

        p_load  = float(env.p_load_series[t])
        p_solar = float(env.p_solar_series[t])

        # Update env internal history so subsequent _get_obs() sees lag features
        env._load_history.append(p_load)
        env._solar_history.append(p_solar)
        env._soc_history.append(soc)
        env.t += 1

        # Run load flow (same as rule-based path)
        lf   = run_load_flow(env.net, p_load, p_solar)
        label = generate_blackout_label(
            p_load, p_solar, p_battery, lf['v_min'], lf['converged']
        )
        severity = generate_severity_label(
            p_load, p_solar, p_battery,
            lf['v_min'], lf['converged'], lf['line_loading_max']
        )
        p_grid = p_load - p_solar - p_battery

        records.append({
            'timestep':         t,
            'hour_of_day':      t % 24,
            'day':              t // 24,
            'p_load_mw':        round(p_load,                6),
            'p_solar_mw':       round(p_solar,               6),
            'p_battery_mw':     round(p_battery,             6),
            'p_grid_mw':        round(p_grid,                6),
            'soc':              round(soc,                   6),
            'v_min':            round(lf['v_min'],           6),
            'v_max':            round(lf['v_max'],           6),
            'v_mean':           round(lf['v_mean'],          6),
            'line_loading_max': round(lf['line_loading_max'],4),
            'v_severity':       round(lf['v_severity'],      6),
            'n_buses_below_090':lf['n_buses_below_090'],
            'renewable_pct':    round(lf['renewable_pct'],   2),
            'converged':        int(lf['converged']),
            'blackout':         label,
            'severity':         severity,
        })

        if t % 100 == 0:
            print(f"  [RL] t={t}/{T} | Load={p_load:.3f} MW | "
                  f"Solar={p_solar:.3f} MW | SOC={soc:.3f} | "
                  f"Vmin={lf['v_min']:.4f} | Blackout={label}")

    env.close()
    return pd.DataFrame(records)


# ── Phase 1: Run full physics simulation ──────────────────────────────────────

def run_physics_simulation(
    load_path: str,
    solar_path: str,
    n_days: int = 30,
    battery_capacity: float = 2.0,
    battery_soc_init: float = 0.5,
) -> pd.DataFrame:
    """
    Run the full physics simulation using real data.

    This replicates real grid conditions hour-by-hour using:
      - Real electricity consumption patterns (UCI dataset)
      - Real solar generation data (Kaggle Plant 1)
      - IEEE 33-bus Newton-Raphson AC power flow
      - Physics-constrained battery model

    Returns a DataFrame where each row = one hour of real grid state.
    """
    print("[1/3] Loading real datasets...")
    p_load_series  = prepare_load_series(load_path, n_days)
    p_solar_series = prepare_solar_series(solar_path, n_days)

    T = min(len(p_load_series), len(p_solar_series))
    p_load_series  = p_load_series.iloc[:T]
    p_solar_series = p_solar_series.iloc[:T]
    print(f"       Data loaded: {T} hours ({T//24} days)")
    print(f"       Load  : {p_load_series.min():.3f} – {p_load_series.max():.3f} MW")
    print(f"       Solar : {p_solar_series.min():.3f} – {p_solar_series.max():.3f} MW")

    battery = BatteryModel(
        capacity_mwh=battery_capacity,
        soc_init=battery_soc_init
    )
    net = create_ieee33_network()
    print(f"       Battery: {battery.capacity} MWh | Network: {len(net.bus)} buses")

    print(f"[2/3] Running physics simulation ({T} timesteps)...")
    records = []
    for t in range(T):
        p_load  = float(p_load_series.iloc[t])
        p_solar = float(p_solar_series.iloc[t])

        # Battery dispatch
        p_battery, soc = battery.step(p_load, p_solar)

        # AC load flow — the heart of the digital twin
        lf = run_load_flow(net, p_load, p_solar)

        p_grid = p_load - p_solar - p_battery

        # Ground truth physics labels
        blackout = generate_blackout_label(
            p_load, p_solar, p_battery, lf['v_min'], lf['converged']
        )
        severity = generate_severity_label(
            p_load, p_solar, p_battery,
            lf['v_min'], lf['converged'], lf['line_loading_max']
        )

        records.append({
            'timestep':         t,
            'hour_of_day':      t % 24,
            'day':              t // 24,
            'p_load_mw':        round(p_load,                 6),
            'p_solar_mw':       round(p_solar,                6),
            'p_battery_mw':     round(p_battery,              6),
            'p_grid_mw':        round(p_grid,                 6),
            'soc':              round(soc,                    6),
            'v_min':            round(lf['v_min'],            6),
            'v_max':            round(lf['v_max'],            6),
            'v_mean':           round(lf['v_mean'],           6),
            'line_loading_max': round(lf['line_loading_max'], 4),
            'v_severity':       round(lf['v_severity'],       6),
            'n_buses_below_090':lf['n_buses_below_090'],
            'renewable_pct':    round(lf['renewable_pct'],    2),
            'converged':        int(lf['converged']),
            'blackout':         blackout,      # PHYSICS GROUND TRUTH
            'severity':         severity,
        })

        if (t + 1) % 24 == 0:
            day_recs = records[-24:]
            bo_hrs = sum(r['blackout'] for r in day_recs)
            avg_v  = sum(r['v_min'] for r in day_recs) / 24
            print(f"  Day {t//24+1:3d}/{T//24} | "
                  f"BO={bo_hrs:2d}h | "
                  f"Avg V_min={avg_v:.4f} | "
                  f"SOC={soc:.3f} | "
                  f"Load={p_load:.3f} MW")

    print(f"[3/3] Adding temporal features...")
    df = pd.DataFrame(records)
    df = add_temporal_features(df)
    print(f"       Total blackout hours : {df['blackout'].sum()} / {T} "
          f"({df['blackout'].mean()*100:.1f}%)")
    print(f"       Min V_min observed   : {df['v_min'].min():.4f} pu")
    print(f"       Avg renewable pct    : {df['renewable_pct'].mean():.1f}%")
    return df


# ── Phase 2: Train ML predictor on first 80% ─────────────────────────────────

def train_predictor(df: pd.DataFrame):
    """
    Train the ML predictor on the first 80% of simulation data.

    Forecasting objective
    ---------------------
    The model predicts blackout at hour t+1 given the grid state at hour t.
    This is the realistic operating scenario: an operator observes the
    current state and wants to know if the NEXT hour will be a blackout.

    Using same-timestep features (X[t] -> y[t]) achieves ROC-AUC ≈ 1.0
    because the label is a deterministic function of the features (e.g.
    blackout <- p_load > p_solar + p_battery + 3.5 MW). That is not ML,
    it is rediscovering the physics formula.

    With 1-step-ahead prediction (X[t] -> y[t+1]) the model must truly
    generalise from patterns — it cannot know next hour's load-flow output.

    Overfitting safeguards
    ----------------------
    - Train vs test score comparison (gap > 0.15 = warning)
    - Per-class precision/recall reported (exposes bias towards majority class)
    - Feature importance printed for Random Forest (catches data leakage)
    - Class imbalance ratio printed

    Returns
    -------
    (model, scaler, split_idx, feat_cols)  or  (None, None, split_idx, None)
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, classification_report, f1_score,
        precision_score, recall_score, average_precision_score,
    )

    feat_cols = [f for f in FEATURE_COLS if f in df.columns]

    # ── 1-step-ahead target shift ─────────────────────────────────────────
    # X[t] -> y[t+1]: drop the last row (no future label), drop first row of y.
    X_all = df[feat_cols].values[:-PREDICT_AHEAD]      # features at t
    y_all = df['blackout'].values[PREDICT_AHEAD:]      # label   at t+1

    X = X_all
    y = y_all
    print(f"\n  Forecasting horizon : {PREDICT_AHEAD} hour(s) ahead")
    print(f"  Samples after shift : {len(X)} (original {len(df)} - {PREDICT_AHEAD})")

    n_total_pos = int(y.sum())

    # ── Guard: no events at all ───────────────────────────────────────────
    if n_total_pos == 0:
        print("\n  [ERROR] Zero blackout events in entire dataset.")
        print("          Check label_generator.py thresholds vs data range.")
        return None, None, int(len(X)*0.8), None

    # ── Smart split: guarantee blackouts in both train AND test ───────────
    # A rigid temporal 80/20 split on 30 days often puts ALL blackouts in
    # the training window, leaving the test set empty — then ROC-AUC cannot
    # be computed and classification_report crashes with mismatched classes.
    # We try temporal first, then fall back to stratified if needed.
    min_test_pos = max(3, int(n_total_pos * 0.2))

    split_idx = int(len(X) * 0.8)
    X_train_t, X_test_t = X[:split_idx], X[split_idx:]
    y_train_t, y_test_t = y[:split_idx], y[split_idx:]

    if y_test_t.sum() >= min_test_pos:
        X_train, X_test = X_train_t, X_test_t
        y_train, y_test = y_train_t, y_test_t
        split_strategy = 'temporal 80/20'
    else:
        from sklearn.model_selection import train_test_split as sk_split
        X_train, X_test, y_train, y_test = sk_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        split_strategy = 'stratified (temporal had 0 test blackouts)'

    n_train_pos = int(y_train.sum());  n_train_neg = len(y_train) - n_train_pos
    n_test_pos  = int(y_test.sum());   n_test_neg  = len(y_test)  - n_test_pos

    print(f"\n  Split strategy: {split_strategy}")
    print(f"  Training set : {len(X_train)} hours "
          f"| Normal: {n_train_neg} | Blackout: {n_train_pos} "
          f"({y_train.mean()*100:.1f}%)")
    print(f"  Test set     : {len(X_test)} hours "
          f"| Normal: {n_test_neg} | Blackout: {n_test_pos} "
          f"({y_test.mean()*100:.1f}%)")

    if n_train_pos == 0:
        print(f"\n  [ERROR] No blackout events in training set after split.")
        print(f"          Try increasing n_days in main.py.")
        return None, None, split_idx, None

    # ── Guard: extreme imbalance ──────────────────────────────────────────
    imbalance_ratio = n_train_neg / max(n_train_pos, 1)
    if imbalance_ratio > 20:
        print(f"\n  [WARNING] Severe class imbalance: {imbalance_ratio:.0f}:1")
        print(f"            Using class_weight='balanced' to compensate.")
    elif imbalance_ratio > 5:
        print(f"\n  [INFO] Moderate imbalance: {imbalance_ratio:.1f}:1")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Candidate models ──────────────────────────────────────────────────
    candidates = {
        'RandomForest': (
            RandomForestClassifier(
                n_estimators=200, max_depth=8,
                class_weight='balanced',
                random_state=42, n_jobs=-1
            ),
            False   # uses unscaled features
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, random_state=42
            ),
            False
        ),
        'LogisticRegression': (
            LogisticRegression(
                class_weight='balanced',
                max_iter=1000, random_state=42
            ),
            True    # needs scaled features
        ),
    }

    best_f1, best_roc, best_name, best_model = 0, 0, None, None
    results = {}
    test_has_positives = n_test_pos > 0

    if not test_has_positives:
        print(f"\n  [WARNING] Test set has 0 blackout events (all {n_test_pos+n_test_neg} normal).")
        print(f"            ROC-AUC on test cannot be computed.")
        print(f"            Selecting best model by train ROC-AUC instead.")
        print(f"            Consider using n_days > 30 for better evaluation.")

    print(f"\n  {'Model':<22} {'Train-ROC':>10} {'Test-ROC':>9} "
          f"{'Gap':>6} {'Test-F1':>8} {'Prec':>6} {'Recall':>7}  Status")
    print(f"  {'-'*85}")

    for name, (m, scaled) in candidates.items():
        Xtr = X_train_s if scaled else X_train
        Xte = X_test_s  if scaled else X_test

        m.fit(Xtr, y_train)

        # ── Train score (for overfitting detection) ───────────────────────
        yp_train = m.predict(Xtr)
        yprob_train = m.predict_proba(Xtr)[:, 1]
        try:
            roc_train = roc_auc_score(y_train, yprob_train)
        except ValueError:
            roc_train = 0.0

        # ── Test score ────────────────────────────────────────────────────
        yp    = m.predict(Xte)
        yprob = m.predict_proba(Xte)[:, 1]
        try:
            roc_test = roc_auc_score(y_test, yprob)
        except ValueError:
            roc_test = 0.0
        f1   = f1_score(y_test, yp, zero_division=0)
        prec = precision_score(y_test, yp, zero_division=0)
        rec  = recall_score(y_test, yp, zero_division=0)

        # -- Overfitting check -------------------------------------------------
        gap    = roc_train - roc_test if test_has_positives else 0.0
        status = ''
        if not test_has_positives:
            status = '[NO TEST POS]'
        elif gap > 0.15:
            status = '[OVERFIT]'
        elif gap > 0.08:
            status = '[MILD OVERFIT]'
        elif roc_test < 0.55:
            status = '[WEAK]'
        else:
            status = 'OK'

        # -- Bias check: does it just predict majority class? ------------------
        pred_pos_rate = yp.mean()
        if pred_pos_rate < 0.01 and y_test.mean() > 0.05:
            status += ' [ALL-NEG BIAS]'
        elif pred_pos_rate > 0.99 and y_test.mean() < 0.95:
            status += ' [ALL-POS BIAS]'

        results[name] = {
            'roc_train': roc_train, 'roc_test': roc_test,
            'f1': f1, 'precision': prec, 'recall': rec, 'gap': gap,
        }
        print(f"  {name:<22} {roc_train:>10.4f} {roc_test:>9.4f} "
              f"{gap:>+6.3f} {f1:>8.4f} {prec:>6.3f} {rec:>7.3f}  {status}")

        # Select best model: by test ROC if available, else by train ROC
        if test_has_positives:
            if roc_test > best_roc:
                best_roc, best_name, best_model = roc_test, name, m
        else:
            if roc_train > best_roc:
                best_roc, best_name, best_model = roc_train, name, m

    # -- Best model details ------------------------------------------------
    roc_label = 'test' if test_has_positives else 'train'
    print(f"\n  Best model: {best_name} (ROC-AUC {roc_label}={best_roc:.4f})")

    Xte = X_test_s if 'Logistic' in best_name else X_test
    yp  = best_model.predict(Xte)
    print(f"\n  Classification report ({best_name} on TEST set):")
    # Guard: classification_report crashes if test set has only one class
    unique_classes = np.unique(np.concatenate([y_test, yp]))
    if len(unique_classes) < 2:
        present_names = ['Normal'] if 0 in unique_classes else ['Blackout']
        print(classification_report(y_test, yp,
                                    target_names=present_names,
                                    zero_division=0))
        print("  [NOTE] Only one class present in test set — full report unavailable.")
    else:
        print(classification_report(y_test, yp,
                                    target_names=['Normal', 'Blackout'],
                                    zero_division=0))

    # ── Feature importance (Random Forest only) ───────────────────────────
    if best_name == 'RandomForest' and hasattr(best_model, 'feature_importances_'):
        imp = best_model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:8]
        print(f"  Top 8 feature importances:")
        for rank, i in enumerate(top_idx, 1):
            print(f"    {rank}. {feat_cols[i]:<25} {imp[i]:.4f}")
        print()

    # -- Overfitting summary -----------------------------------------------
    gap = results[best_name]['gap']
    if not test_has_positives:
        print(f"  [INFO] Cannot assess overfitting -- test set has no blackouts.")
        print(f"         Train ROC = {results[best_name]['roc_train']:.4f}")
    elif gap > 0.15:
        print(f"  [!!] OVERFITTING DETECTED (train-test gap = {gap:.3f})")
        print(f"       The model memorises training data but fails on new data.")
        print(f"       Consider: fewer features, lower max_depth, more data.")
    elif gap > 0.08:
        print(f"  [!] Mild overfitting (gap = {gap:.3f}). Monitor carefully.")
    else:
        print(f"  [OK] No significant overfitting (gap = {gap:.3f})")

    # -- Save --------------------------------------------------------------
    os.makedirs('outputs/model', exist_ok=True)
    joblib.dump(best_model, 'outputs/model/best_model.pkl')
    joblib.dump(scaler,     'outputs/model/scaler.pkl')
    joblib.dump(feat_cols,  'outputs/model/feature_cols.pkl')
    print(f"\n  Saved: outputs/model/best_model.pkl")

    # LSTM if PyTorch available
    try:
        from ml_model import train_lstm_model
        train_lstm_model(X_train_s, y_train, X_test_s, y_test)
    except Exception:
        pass

    return best_model, scaler, split_idx, feat_cols


# ── Phase 3: Apply predictor across full timeline ─────────────────────────────

def apply_predictor(df: pd.DataFrame, model, scaler, feat_cols: list,
                    split_idx: int) -> pd.DataFrame:
    """
    Apply the trained ML predictor in 1-step-ahead forecasting mode.

    blackout_prob_ml[t] = P(blackout at t+1 | state at t)
    The first PREDICT_AHEAD rows get NaN (no prior state to predict from).

    Two parallel signals:
      blackout         -- physics ground truth at t (what happened)
      blackout_prob_ml -- ML forecast made at t-1   (what was predicted)
    """
    df = df.copy()
    feat_cols = [f for f in feat_cols if f in df.columns]

    # Features at t -> predict blackout at t+1
    X = df[feat_cols].values[:-PREDICT_AHEAD]   # rows 0 ... T-2

    use_scaled = hasattr(model, 'coef_')         # LogisticRegression needs scaling
    X_input = scaler.transform(X) if (scaler is not None and use_scaled) else X

    probs = model.predict_proba(X_input)[:, 1]

    # Pad with NaN for the first row (no prediction available yet)
    nan_pad = np.full(PREDICT_AHEAD, np.nan)
    probs_full = np.concatenate([nan_pad, probs])

    df['blackout_prob_ml'] = np.round(probs_full, 6)
    df['blackout_pred_ml'] = np.where(
        np.isnan(probs_full), 0, (probs_full >= 0.5).astype(int)
    )
    df['in_test_window'] = (df.index >= split_idx).astype(int)

    # Did the forecast (made at t-1) correctly predict the blackout at t?
    df['prediction_correct'] = (
        df['blackout_pred_ml'] == df['blackout']
    ).astype(int)
    return df




# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_digital_twin(
    load_path: str,
    solar_path: str,
    n_days: int = 30,
    battery_capacity: float = 2.0,
    battery_soc_init: float = 0.5,
    use_rl: bool = False,
    rl_policy_path: str = 'outputs/rl_policy/best_model',
) -> pd.DataFrame:
    """
    Full digital twin pipeline.

    1. Physics simulation: replicate real grid conditions from real data
    2. ML training: learn to predict blackout from grid state (first 80%)
    3. Prediction: apply predictor across full timeline (both windows)
    4. Save: CSV with physics truth + ML predictions side by side

    Returns
    -------
    pd.DataFrame with all physics state, labels, and ML predictions.
    """
    print("="*60)
    print("  DIGITAL TWIN — Physics Simulation")
    print("="*60)

    # ── RL path: swap in trained policy for battery dispatch ──────────────
    if use_rl:
        df = _run_rl_simulation(
            load_path, solar_path, n_days,
            battery_capacity, rl_policy_path
        )
    else:
        df = run_physics_simulation(
            load_path, solar_path, n_days,
            battery_capacity, battery_soc_init
        )

    print("\n" + "=" * 60)
    print("  DIGITAL TWIN — ML Predictor Training")
    print("=" * 60)
    result = train_predictor(df)
    if result[0] is None:
        print("[ERROR] Could not train predictor (no blackout events).")
        df.to_csv('outputs/simulation_results.csv', index=False)
        return df

    model, scaler, split_idx, feat_cols = result

    print("\n" + "=" * 60)
    print("  DIGITAL TWIN — Applying Predictor Across Timeline")
    print("=" * 60)
    df = apply_predictor(df, model, scaler, feat_cols, split_idx)

    # Agreement stats
    test_df = df[df['in_test_window'] == 1]
    if len(test_df) > 0:
        acc = test_df['prediction_correct'].mean()
        print(f"\n  Out-of-sample accuracy  : {acc*100:.2f}%")
        bo_caught = test_df[test_df['blackout']==1]['blackout_pred_ml'].sum()
        bo_total  = test_df['blackout'].sum()
        if bo_total > 0:
            recall = bo_caught / bo_total
            print(f"  Blackout recall (test)  : {recall*100:.1f}% "
                  f"({int(bo_caught)}/{int(bo_total)} events caught)")

    # Save
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/simulation_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Key columns: blackout (physics truth) | "
          f"blackout_prob_ml (ML prediction) | in_test_window")

    # Final summary
    T = len(df)
    bo = df['blackout'].sum()
    eens = df.loc[df['blackout']==1, 'p_load_mw'].sum()
    lolp = bo / T

    print(f"\n{'='*60}")
    print(f"  DIGITAL TWIN SUMMARY")
    print(f"{'='*60}")
    sev = df['severity'].value_counts().sort_index()
    print(f"  Hours simulated         : {T}")
    print(f"  Blackout hours          : {bo} ({lolp*100:.2f}%)")
    print(f"  LOLP                    : {lolp:.4f}")
    print(f"  EENS                    : {eens:.3f} MWh")
    print(f"  Severity breakdown:")
    print(f"    Normal   (0) : {sev.get(0,0)} hours")
    print(f"    Warning  (1) : {sev.get(1,0)} hours")
    print(f"    Critical (2) : {sev.get(2,0)} hours")
    print(f"    Blackout (3) : {sev.get(3,0)} hours")
    print(f"  Training window (h 1–{split_idx}) : physics simulation only")
    print(f"  Test window (h {split_idx}–{T})   : ML predicts BEFORE physics label")
    print(f"{'='*60}")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    LOAD_PATH  = r'datasets\LD2011_2014.txt'
    SOLAR_PATH = r'datasets\Plant_1_Generation_Data.csv'

    df = run_digital_twin(LOAD_PATH, SOLAR_PATH, n_days=30)
    print(f"\nFirst 5 rows with predictions:")
    cols = ['timestep','hour_of_day','p_load_mw','p_solar_mw','v_min',
            'blackout','severity','blackout_prob_ml','blackout_pred_ml','in_test_window']
    print(df[cols].head(10).to_string())
