# Digital Twin Microgrid — Complete Project Explanation

---

## What This Project Is

This project is a **software simulation** of a small power grid (microgrid). There is no real hardware — everything is simulated in Python.

The simulation mimics what happens in a real microgrid **hour by hour**:
- How much power is consumed (load)
- How much solar is generating
- Whether the battery helps
- Whether the grid is stressed enough to cause a **blackout**

Then an **ML model** is trained to **predict the probability of a blackout** given the system state.

---

## The Real-World Analogy

Think of a small residential colony with:
- **Solar panels** on rooftops generating power during the day
- **A battery bank** that stores excess solar and discharges at night
- **A main grid connection** as backup
- **33 distribution lines** and buses connecting all consumers (IEEE 33-bus network)

Your job is to simulate this colony for **30 days** and find out **when it is at risk of a blackout**.

---

## Full Pipeline in One Picture

```
UCI Load CSV
      +           →  data_load.py   →  p_load(t)  ──┐
Kaggle Solar CSV                    →  p_solar(t)  ──┤
                                                      │
                                              battery_model.py
                                                      │
                                              p_battery(t), soc(t)
                                                      │
                                              grid_simulator.py
                                           (pandapower IEEE 33-bus)
                                                      │
                                              v_min(t), v_max(t)
                                                      │
                                              label_generator.py
                                                      │
                                              blackout(t) = 0 or 1
                                                      │
                                              ml_model.py
                                                      │
                                           P(blackout) = 0.0 to 1.0
                                                      │
                                              Streamlit Dashboard
```

---

## File-by-File Explanation

---

### 1. `data_load.py` — Load Data Preprocessing

**Purpose:** Converts raw UCI Electricity Load data into simulation-ready hourly demand in MW.

**Input:** `datasets/LD2011_2014.txt` — 370 electricity clients, 15-minute resolution, European format.

**What it does step by step:**

```
LD2011_2014.txt
    │
    ▼
read_csv()         → DataFrame: 370 columns (clients), rows = 15-min timestamps
    │                  sep=';', decimal=',' (European CSV format)
    │
sort_index()       → Ensures chronological order
    │
replace('?', NaN)  → UCI uses '?' for missing values, not NaN
    │
ffill().bfill()    → Forward-fill then backward-fill gaps
    │
sum(axis=1)        → Aggregate all 370 clients → total kW per timestamp
    │
÷ 1000             → kW → MW
    │
resample('h')      → 15-min → hourly (average 4 readings per hour)
    │
normalize_by_max() → Scale to [0, 1] — keeps the shape, removes magnitude
    │
× 3.715 MW         → Map onto IEEE 33-bus base load capacity
    │
trim to 30 days    → First 720 hours only
    │
    ▼
p_load(t)          → Ready for simulation
```

**Key functions:**
- `load_electricity_370(path)` — reads raw CSV, handles '?', aggregates, converts
- `normalize_by_max(series)` — divides by max value
- `scale_to_ieee_base(series, 3.715)` — maps to IEEE 33-bus capacity
- `prepare_load_series(path, n_days=30)` — chains all steps together
- `validate_load_series(series)` — checks for NaN, negatives, capacity breach

**Why normalization?** The UCI dataset has 370 clients totalling ~500 MW peak. The IEEE 33-bus handles only 3.715 MW. We borrow the **shape** of real data (daily patterns), not the actual values.

---

### 2. `data_solar.py` — Solar Data Preprocessing

**Purpose:** Converts Kaggle Solar Power Generation data into simulation-ready hourly PV generation in MW.

**Input:** `datasets/Plant_1_Generation_Data.csv` — 22 inverters, 15-minute resolution, ~34 days.

**What it does step by step:**

```
Plant_1_Generation_Data.csv
    │
    ▼
read_csv()              → DataFrame with DATE_TIME, AC_POWER, etc.
    │
parse dates             → DD-MM-YYYY HH:MM format
    │
groupby(DATE_TIME).sum()→ Sum AC_POWER across all 22 inverters
    │
fillna(0)               → Night hours = 0 (no solar)
    │
÷ 1000                  → kW → MW
    │
resample('h').mean()    → 15-min → hourly
    │
normalize_by_max()      → Scale to [0, 1]
    │
× 2.0 MW               → Map onto PV capacity
    │
trim to 30 days         → First 720 hours
    │
    ▼
p_solar(t)              → Ready for simulation
```

**Key difference from load data:**
- Missing values are filled with 0.0 (correct — no solar at night)
- Groups by timestamp first (multiple inverters per timestamp)

---

### 3. `battery_model.py` — Battery Energy Storage

**Purpose:** Simulates a rule-based battery that charges from solar surplus and discharges to cover load deficit.

**Parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| Capacity | 2.0 MWh | Total energy storage |
| SOC init | 0.5 (50%) | Starting charge level |
| SOC min | 0.1 (10%) | Never fully drain (protects battery) |
| SOC max | 0.9 (90%) | Never fully charge (protects battery) |
| Charge rate | 0.5 MW | Max charging power |
| Discharge rate | 0.5 MW | Max discharging power |
| Efficiency | 0.95 (95%) | Energy lost in each direction |

**How `step(p_load, p_solar)` works:**

```
surplus = p_solar - p_load

IF surplus > 0 (solar exceeds load):
    → CHARGE battery
    → p_charge = min(surplus, charge_rate, headroom)
    → SOC increases
    → p_battery = -p_charge (negative = absorbing)

IF surplus <= 0 (load exceeds solar):
    → DISCHARGE battery
    → p_discharge = min(deficit, discharge_rate, available_energy)
    → SOC decreases
    → p_battery = +p_discharge (positive = supplying)
```

**Three constraints limit power at each step:**
1. **Charge/discharge rate** — inverter can't push more than 0.25 MW
2. **Headroom** — can't charge beyond 90% SOC
3. **Available energy** — can't discharge below 10% SOC

**Efficiency (η = 0.95):**
- Charging: put in 0.25 MW → battery stores 0.25 × 0.95 = 0.2375 MWh
- Discharging: battery gives 0.25 MW → loses 0.25 / 0.95 = 0.263 MWh from SOC
- Battery drains slightly faster than it charges — 5% lost every cycle

---

### 4. `grid_simulator.py` — IEEE 33-Bus Power Flow

**Purpose:** Runs AC load flow simulation on the IEEE 33-bus test network using `pandapower`.

**What is IEEE 33-bus?**
A standard benchmark power distribution network with:
- 33 buses (nodes)
- 32 branches (lines)
- 1 slack bus (main grid connection)
- Well-validated, reproducible, used in hundreds of papers

**What the function does:**

```
At each timestep t:
    1. Load IEEE 33-bus network (case33bw)
    2. Scale all bus loads proportionally:
       - base loads sum to 3.715 MW
       - multiply by (p_load / 3.715) to match current demand
    3. Inject PV at bus 17 (mid-feeder) as static generator
    4. Run Newton-Raphson AC power flow
    5. Extract results:
       - v_min: minimum voltage across all 33 buses (pu)
       - v_max: maximum voltage
       - v_mean: average voltage
       - line_loading_max: most loaded line (%)
       - converged: did the load flow solve?
```

**Why AC load flow (not DC)?**
DC approximation ignores reactive power and voltage. AC captures both — essential for detecting voltage violations that cause blackouts.

**Why bus 17?**
Mid-feeder injection point. PV at the middle of the feeder has the most realistic impact on voltage profiles.

---

### 5. `label_generator.py` — Blackout Labels

**Purpose:** Generates physics-based binary labels (0 = normal, 1 = blackout).

**Three conditions trigger a blackout (ANY one is enough):**

| Condition | Check | Threshold |
|-----------|-------|-----------|
| Power deficit | `p_grid_required > limit` | Grid import > 3.5 MW |
| Voltage violation | `v_min < threshold` | Min bus voltage < 0.92 pu |
| Non-convergence | `converged == False` | Load flow failed to solve |

**The formula:**
```python
p_grid_required = p_load - p_solar - p_battery
blackout = (p_grid_required > 3.5) OR (v_min < 0.92) OR (not converged)
```

**Why physics-based labels?**
The ground truth comes from real power system constraints, not arbitrary human annotation. This makes the project academically valid — the ML model learns from physics, not guesswork.

---

### 6. `digital_twin.py` — Main Simulation Loop

**Purpose:** Connects everything into a single timestep-by-timestep simulation.

**What it does:**

```
FOR each hour t = 0, 1, 2, ... 719:

    1. p_load(t)   ← from data_load.py
    2. p_solar(t)  ← from data_solar.py
    3. battery.step(p_load, p_solar) → p_battery(t), soc(t)
    4. run_load_flow(net, p_load, p_solar) → v_min, v_max, line_loading
    5. p_grid(t) = p_load - p_solar - p_battery
    6. generate_blackout_label(...) → blackout = 0 or 1
    7. Record all values as one row

SAVE all 720 rows → outputs/simulation_results.csv
```

**Output CSV columns:**
```
timestep | hour_of_day | day | p_load_mw | p_solar_mw | p_battery_mw |
p_grid_mw | soc | v_min | v_max | v_mean | line_loading_max |
converged | blackout
```

This CSV is the **training dataset** for the ML model.

---

### 7. `ml_model.py` — Machine Learning Prediction

**Purpose:** Trains classifiers to predict blackout probability from the simulation data.

**Features used (11 base + 5 engineered = 16 total):**

Base features:
- `p_load_mw`, `p_solar_mw`, `p_battery_mw`, `p_grid_mw`
- `soc`, `v_min`, `v_max`, `v_mean`, `line_loading_max`
- `hour_of_day`, `converged`

Engineered features:
- `solar_load_ratio` = solar / load (PV coverage)
- `soc_deficit` = 1.0 - soc (battery depletion)
- `v_violation_margin` = v_min - 0.95 (negative = violation)
- `net_power_balance` = solar + battery - load (surplus/deficit)
- `is_night` = 1 if hour < 6 or hour > 20

**Three models trained:**

| Model | Why |
|-------|-----|
| RandomForest | Handles tabular data well, interpretable feature importances |
| GradientBoosting | Often best accuracy on structured data |
| LogisticRegression | Simple baseline, needs feature scaling |

**Metrics:**
- **ROC-AUC** — area under ROC curve (higher = better)
- **Average Precision** — useful for imbalanced classes
- **Brier Score** — calibration of probability estimates (lower = better)

**Reliability Metrics:**
- **LOLP** (Loss of Load Probability) = fraction of hours with blackout
- **EENS** (Expected Energy Not Served) = total MWh of load during blackouts

**Output:** Best model saved to `outputs/model/best_model.pkl`

---

### 8. `main.py` — Entry Point

**Purpose:** Runs the full pipeline in order.

```
Step 1: digital_twin.py → simulation_results.csv
Step 2: ml_model.py     → best_model.pkl + metrics
Step 3: ml_model.py     → LOLP + EENS
```

**To run:** `python main.py`

---

### 9. `dashboard/app.py` — Streamlit Dashboard (Optional)

**Purpose:** Interactive web visualization of simulation results.

**To run:** `streamlit run dashboard/app.py`

Shows:
- KPI metrics (timesteps, blackout hours, LOLP%, min voltage)
- Power profiles (load, solar, battery) with blackout overlay
- Voltage profile with 0.95 pu violation line
- Battery SOC with min/max bounds
- Probabilistic blackout risk from ML model

---

## Key Concepts for Viva

| Question | Answer |
|----------|--------|
| Why IEEE 33-bus? | Standard benchmark, well-validated, reproducible |
| Why AC load flow over DC? | Captures reactive power and voltage — essential for blackout detection |
| Why physics-based labels? | Ground truth from real power constraints, not arbitrary annotation |
| What does LOLP mean? | Loss of Load Probability — fraction of time system fails to meet demand |
| Why RandomForest over deep learning? | Interpretable, handles tabular data well,limited data doesn't justify DL |
| Why probabilistic output? | Risk quantification (P=0.87) is more useful than binary (yes/no) for operations |
| What is the algorithm name? | PHYSIC-DT-RISK — physics-informed simulation driving ML prediction |
| Why normalize real data? | Real data is 500+ MW, IEEE 33-bus handles 3.715 MW. We borrow the shape, not the scale |
| Why 30 days? | Solar data only covers 34 days. 720 hours is enough for ML training |
| How does the battery protect against blackouts? | Stores solar surplus during day, discharges at night to reduce grid stress |

---

## How to Run

```bash
# Step 1: Run the full pipeline
python main.py

# Step 2 (optional): Launch dashboard
streamlit run dashboard/app.py
```

---

## Project Structure

```
archive/
├── datasets/
│   ├── LD2011_2014.txt              # UCI load data
│   └── Plant_1_Generation_Data.csv  # Kaggle solar data
├── outputs/
│   ├── simulation_results.csv       # 720 rows of simulation data
│   └── model/
│       ├── best_model.pkl           # Trained ML model
│       └── scaler.pkl               # Feature scaler
├── dashboard/
│   └── app.py                       # Streamlit dashboard
├── data_load.py                     # Load preprocessing
├── data_solar.py                    # Solar preprocessing
├── battery_model.py                 # Battery model
├── grid_simulator.py                # IEEE 33-bus load flow
├── label_generator.py               # Blackout label generator
├── digital_twin.py                  # Main simulation loop
├── ml_model.py                      # ML training & evaluation
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
└── 01_load_processing.ipynb         # Load data exploration notebook
```
