"""
Label Generator for Digital Twin Microgrid.

Generates physics-based binary blackout labels using
power balance, voltage constraints, and load flow convergence.

Key design choices
──────────────────
Voltage threshold: V_MIN = 0.92 pu
  → IEEE 33-bus at base load (3.715 MW) produces v_min ≈ 0.913 pu
  → At typical load (~2.5 MW) v_min ≈ 0.94 pu
  → 0.92 pu captures moderate-to-heavy stress events realistically
  → Results in LOLP of 5–20% for a stressed microgrid

Grid import limit: 3.5 MW
  → The feeder peak is 3.715 MW so the grid connection is tight
  → Night-time peaks without DER support trigger power deficit
  → This creates a meaningful role for battery dispatch (RL target)

Enhancement: severity score
  → generate_blackout_label()   returns binary 0/1 as before
  → generate_severity_label()   returns 0/1/2/3 multi-class label
      0 = Normal (all OK)
      1 = Warning (voltage 0.88–0.90 OR line loading 80–100%)
      2 = Critical (voltage 0.85–0.88 OR line loading > 100%)
      3 = Blackout (voltage < 0.85 OR power deficit OR non-convergence)
"""


# ── Thresholds ────────────────────────────────────────────────────────────────

# Voltage limits (tuned for IEEE 33-bus with load range 1.4–3.7 MW)
V_MIN_THRESHOLD      = 0.92    # Moderate voltage dip → stress event
V_WARNING_THRESHOLD  = 0.90    # Approaching emergency — raise alert
V_EMERGENCY_THRESHOLD= 0.87    # Emergency limit — severe blackout risk

# Grid import limits (tight — forces DER/battery to contribute)
GRID_IMPORT_LIMIT_MW = 3.5     # Maximum grid import before power deficit
LINE_WARNING_PCT     = 80.0    # Line loading warning threshold (%)
LINE_CRITICAL_PCT    = 100.0   # Line loading critical threshold (%)


def generate_blackout_label(
    p_load: float,
    p_solar: float,
    p_battery: float,
    v_min: float,
    converged: bool,
) -> int:
    """
    Generate a physics-based binary blackout label.

    A blackout (label=1) is triggered if ANY of:
        1. Power deficit   — grid import exceeds 3.5 MW capacity
        2. Voltage violation — v_min drops below 0.92 pu
        3. Non-convergence — AC load flow failed to solve

    Parameters
    ----------
    p_load    : float — total load demand (MW)
    p_solar   : float — PV generation (MW)
    p_battery : float — battery power; +ve = discharge, -ve = charge (MW)
    v_min     : float — minimum bus voltage from load flow (pu)
    converged : bool  — whether AC load flow converged

    Returns
    -------
    int : 1 = blackout, 0 = normal
    """
    p_grid_required = p_load - p_solar - p_battery

    power_deficit     = p_grid_required > GRID_IMPORT_LIMIT_MW
    voltage_violation = v_min < V_MIN_THRESHOLD   # 0.92 pu
    non_convergence   = not converged

    return int(power_deficit or voltage_violation or non_convergence)


def generate_severity_label(
    p_load: float,
    p_solar: float,
    p_battery: float,
    v_min: float,
    converged: bool,
    line_loading_max: float = 0.0,
) -> int:
    """
    Multi-class severity label for richer ML training signal.

    Severity levels
    ───────────────
    0 — Normal   : all constraints satisfied
    1 — Warning  : approaching limits (v_min 0.90–0.92 OR line 80–100%)
    2 — Critical : near emergency   (v_min 0.87–0.90 OR line > 100%)
    3 — Blackout : hard constraint violated (v_min < 0.87 OR deficit OR no-conv)

    Parameters
    ----------
    p_load, p_solar, p_battery : float — MW power values
    v_min          : float — minimum bus voltage (pu)
    converged      : bool  — load flow convergence
    line_loading_max: float — most loaded line (%)

    Returns
    -------
    int : 0 / 1 / 2 / 3
    """
    p_grid_required = p_load - p_solar - p_battery

    # Hard blackout conditions → severity 3
    if (not converged
            or p_grid_required > GRID_IMPORT_LIMIT_MW
            or v_min < V_EMERGENCY_THRESHOLD):
        return 3

    # Critical — near emergency → severity 2
    if v_min < V_WARNING_THRESHOLD or line_loading_max > LINE_CRITICAL_PCT:
        return 2

    # Warning — approaching limits → severity 1
    if v_min < V_MIN_THRESHOLD or line_loading_max > LINE_WARNING_PCT:
        return 1

    # Normal → severity 0
    return 0


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("Label Generator — Quick Test (corrected 0.90 pu threshold)")
    print("=" * 70)

    # Binary label tests
    binary_cases = [
        # (p_load, p_solar, p_battery, v_min, converged, expected)
        (3.0, 1.5, 0.25, 0.97,  True,  0),   # Normal: all OK
        (3.0, 0.0, 0.0,  0.93,  True,  0),   # Night — v_min 0.93 > 0.92: OK
        (3.0, 0.0, 0.0,  0.91,  True,  1),   # Voltage violation (0.91 < 0.92)
        (4.0, 0.0, 0.0,  0.97,  True,  1),   # Power deficit (4.0 > 3.5 MW)
        (3.0, 1.5, 0.25, 0.97,  False, 1),   # Non-convergence
        (5.5, 0.5, 0.1,  0.83,  False, 1),   # All three
    ]

    print(f"\n--- Binary labels (0=normal, 1=blackout) ---")
    print(f"{'#':>2} | {'p_load':>6} | {'p_solar':>7} | {'p_batt':>6} | "
          f"{'v_min':>5} | {'conv':>5} | {'label':>5} | {'expect':>6} | {'OK?':>3}")
    print("-" * 74)

    all_ok = True
    for i, (pl, ps, pb, vm, conv, exp) in enumerate(binary_cases):
        label = generate_blackout_label(pl, ps, pb, vm, conv)
        ok = "✓" if label == exp else "✗"
        if label != exp: all_ok = False
        print(f"{i+1:2d} | {pl:6.1f} | {ps:7.1f} | {pb:6.2f} | "
              f"{vm:5.2f} | {str(conv):>5} | {label:5d} | {exp:6d} | {ok:>3}")
    print(f"\n{'[OK] All binary tests passed.' if all_ok else '[FAIL] Some binary tests failed.'}")

    # Severity label tests
    severity_cases = [
        # (p_load, p_solar, p_battery, v_min, converged, line_pct, expected_sev)
        (2.0, 1.5, 0.25, 0.97,  True,  45.0,  0),   # Normal
        (3.0, 0.5, 0.10, 0.915, True,  82.0,  1),   # Warning: line > 80%
        (3.5, 0.0, 0.10, 0.91,  True,  75.0,  1),   # Warning: voltage 0.90-0.92
        (4.0, 0.0, 0.0,  0.89,  True,  110.0, 2),   # Critical: line > 100%
        (4.5, 0.0, 0.0,  0.88,  True,  95.0,  2),   # Critical: voltage 0.87-0.90
        (6.0, 0.0, 0.0,  0.84,  True,  150.0, 3),   # Blackout: deficit + low V
        (3.0, 1.5, 0.25, 0.97,  False, 0.0,   3),   # Blackout: no convergence
    ]

    print(f"\n--- Severity labels (0=normal, 1=warning, 2=critical, 3=blackout) ---")
    print(f"{'#':>2} | {'v_min':>5} | {'line%':>6} | {'conv':>5} | "
          f"{'sev':>4} | {'expect':>6} | {'OK?':>3}")
    print("-" * 55)

    all_ok2 = True
    for i, (pl, ps, pb, vm, conv, ll, exp) in enumerate(severity_cases):
        sev = generate_severity_label(pl, ps, pb, vm, conv, ll)
        ok = "✓" if sev == exp else "✗"
        if sev != exp: all_ok2 = False
        print(f"{i+1:2d} | {vm:5.2f} | {ll:6.1f} | {str(conv):>5} | "
              f"{sev:4d} | {exp:6d} | {ok:>3}")
    print(f"\n{'[OK] All severity tests passed.' if all_ok2 else '[FAIL] Some severity tests failed.'}")