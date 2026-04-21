"""
rl_environment.py — Gymnasium Environment for Microgrid RL Battery Dispatch.

Wraps the existing digital twin simulation components (battery_model,
grid_simulator, label_generator, data_load, data_solar) to provide a
standard gym.Env interface for RL agents (PPO, SAC, DQN, etc.).

Usage
-----
    from rl_environment import MicrogridEnv
    env = MicrogridEnv(load_path=..., solar_path=..., n_days=30)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from battery_model import BatteryModel
from grid_simulator import create_ieee33_network, run_load_flow
from label_generator import generate_blackout_label
from data_load import prepare_load_series
from data_solar import prepare_solar_series


# ── Environment ───────────────────────────────────────────────────────────────

class MicrogridEnv(gym.Env):
    """
    Gymnasium environment for microgrid battery dispatch optimisation.

    At each timestep the agent chooses a battery action (charge / discharge),
    the digital twin computes the resulting grid state, and a reward is
    returned based on blackout events, voltage quality, and cost.

    Observation Space (12 continuous features, normalised)
    -------------------------------------------------------
        0  soc                  — State of Charge (0-1)
        1  p_load_norm          — Normalised load demand
        2  p_solar_norm         — Normalised solar generation
        3  v_min                — Minimum bus voltage (pu)
        4  hour_norm            — Hour of day / 23
        5  p_load_lag1_norm     — Load 1 hour ago (normalised)
        6  solar_avg6h_norm     — 6-hour solar rolling mean (normalised)
        7  load_avg6h_norm      — 6-hour load rolling mean (normalised)
        8  grid_stress_proxy    — Composite stress indicator (~0-3)
        9  soc_rate_of_change   — SOC change vs. previous step
       10  load_solar_diff_norm — (load - solar) / max_load
       11  line_loading_norm    — Max line loading / 100

    Action Space (continuous)
    --------------------------
        Float in [-1.0, +1.0]
        Negative = charge   (fraction of charge_rate MW)
        Positive = discharge (fraction of discharge_rate MW)
        Zero     = idle

    Reward (composite)
    ------------------
        -10  per blackout event
        +(v_min - 0.92)*5  voltage quality bonus  (or -5 if voltage violation)
        -p_grid * 0.1      grid import cost penalty
        +0.5*soc           SOC health bonus        (or -1 if outside [0.2, 0.8])

    Parameters
    ----------
    load_path : str
        Path to UCI load data (LD2011_2014.txt).
    solar_path : str
        Path to Kaggle solar data (Plant_1_Generation_Data.csv).
    n_days : int
        Simulation horizon in days.
    battery_capacity : float
        Battery capacity in MWh.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        load_path: str,
        solar_path: str,
        n_days: int = 30,
        battery_capacity: float = 2.0,
    ):
        super().__init__()

        # ── Load time-series data ────────────────────────────────────────
        self.p_load_series  = prepare_load_series(load_path, n_days)
        self.p_solar_series = prepare_solar_series(solar_path, n_days)

        T = min(len(self.p_load_series), len(self.p_solar_series))
        self.p_load_series  = self.p_load_series.iloc[:T].values.astype(np.float32)
        self.p_solar_series = self.p_solar_series.iloc[:T].values.astype(np.float32)
        self.T = T

        # ── Battery and network ──────────────────────────────────────────
        self.battery_capacity = battery_capacity
        self.net = create_ieee33_network()

        # ── Normalisation constants ─────────────────────────────────────
        self.max_load  = float(self.p_load_series.max())  or 1.0
        self.max_solar = float(self.p_solar_series.max()) or 1.0

        # ── Gymnasium spaces ─────────────────────────────────────────────
        # All 12 observation features, bounds loosened slightly for robustness
        self.observation_space = spaces.Box(
            low=np.full(12, -1.0, dtype=np.float32),
            high=np.full(12, 3.0, dtype=np.float32),
            dtype=np.float32,
        )

        # Continuous charge/discharge action: [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0],  dtype=np.float32),
            dtype=np.float32,
        )

        # Episode history buffers (for rolling/lag features)
        self._load_history:  list = []
        self._solar_history: list = []
        self._soc_history:   list = []

        # Internal state
        self.battery: BatteryModel = None
        self.t: int = 0

        # Accumulated episode stats
        self._episode_blackouts: int = 0
        self._episode_reward:   float = 0.0


    # ── Gymnasium Interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset to the start of a new episode."""
        super().reset(seed=seed)

        self.battery = BatteryModel(
            capacity_mwh=self.battery_capacity,
            soc_init=0.5,
        )
        self.t = 0
        self._load_history  = []
        self._solar_history = []
        self._soc_history   = []
        self._episode_blackouts = 0
        self._episode_reward    = 0.0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """
        Execute one timestep.

        Parameters
        ----------
        action : np.ndarray, shape (1,)
            Float in [-1, 1]: negative = charge, positive = discharge.

        Returns
        -------
        obs : np.ndarray (12,)
        reward : float
        terminated : bool  (True when episode ends naturally)
        truncated : bool   (always False here)
        info : dict
        """
        assert self.battery is not None, "Call reset() before step()."

        max_rate = self.battery.charge_rate   # 0.25 MW
        p_cmd    = float(action[0]) * max_rate

        p_load  = float(self.p_load_series[self.t])
        p_solar = float(self.p_solar_series[self.t])

        # Apply the RL-commanded battery action (with physics constraints)
        p_battery, soc = self._apply_action(p_cmd)

        # Run AC load flow
        lf = run_load_flow(self.net, p_load, p_solar)

        # Generate blackout label using existing logic
        blackout = generate_blackout_label(
            p_load, p_solar, p_battery, lf['v_min'], lf['converged']
        )

        # Compute reward
        reward = self._compute_reward(
            p_load, p_solar, p_battery, lf['v_min'], blackout, soc
        )

        # Update history buffers
        self._load_history.append(p_load)
        self._solar_history.append(p_solar)
        self._soc_history.append(soc)

        # Advance timestep
        self.t += 1
        self._episode_blackouts += int(blackout)
        self._episode_reward    += reward

        terminated = self.t >= self.T
        truncated  = False

        obs = (
            np.zeros(12, dtype=np.float32)
            if terminated
            else self._get_obs()
        )

        info = {
            'blackout':          blackout,
            'soc':               soc,
            'v_min':             lf['v_min'],
            'p_battery_mw':      p_battery,
            'p_load_mw':         p_load,
            'p_solar_mw':        p_solar,
            'converged':         lf['converged'],
            'episode_blackouts': self._episode_blackouts,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Print a compact one-line summary of the current state."""
        if self.battery is None:
            print("[MicrogridEnv] Not initialised — call reset() first.")
            return
        t       = self.t
        p_load  = float(self.p_load_series[min(t, self.T-1)])
        p_solar = float(self.p_solar_series[min(t, self.T-1)])
        print(
            f"t={t:4d}/{self.T} | "
            f"Load={p_load:.3f} MW | Solar={p_solar:.3f} MW | "
            f"SOC={self.battery.soc:.3f} | "
            f"Blackouts={self._episode_blackouts}"
        )

    def close(self):
        pass


    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _apply_action(self, p_cmd: float):
        """
        Apply an explicit power command to the battery, respecting:
          - charge_rate / discharge_rate inverter limits
          - soc_max / soc_min energy limits

        Parameters
        ----------
        p_cmd : float
            Desired power in MW. Negative = charge, Positive = discharge.

        Returns
        -------
        (p_battery, new_soc)
        """
        if p_cmd < 0:
            # --- CHARGE ---
            headroom = (self.battery.soc_max - self.battery.soc) * self.battery.capacity
            p_charge = min(abs(p_cmd), self.battery.charge_rate, headroom)
            self.battery.soc += (p_charge * self.battery.eta) / self.battery.capacity
            p_battery = -p_charge
        else:
            # --- DISCHARGE ---
            available = (self.battery.soc - self.battery.soc_min) * self.battery.capacity
            p_discharge = min(p_cmd, self.battery.discharge_rate, available)
            self.battery.soc -= p_discharge / (self.battery.capacity * self.battery.eta)
            p_battery = p_discharge

        self.battery.soc = float(np.clip(self.battery.soc, self.battery.soc_min, self.battery.soc_max))
        return p_battery, self.battery.soc

    def _compute_reward(
        self,
        p_load:    float,
        p_solar:   float,
        p_battery: float,
        v_min:     float,
        blackout:  int,
        soc:       float,
    ) -> float:
        """
        Composite reward function.

        Components
        ----------
        1. Blackout penalty   : -10 per blackout event
        2. Voltage quality    : +(v_min-0.92)*5 if OK else -5
        3. Grid import cost   : -p_grid * 0.1
        4. SOC health bonus   : +0.5*soc if in [0.2, 0.8] else -1
        """
        reward = 0.0

        # 1. Blackout — hardest penalty
        if blackout:
            reward -= 10.0

        # 2. Voltage quality
        if v_min >= 0.92:
            reward += (v_min - 0.92) * 5.0
        else:
            reward -= 5.0

        # 3. Grid import cost — encourage self-sufficiency
        p_grid = max(0.0, p_load - p_solar - p_battery)
        reward -= p_grid * 0.1

        # 4. SOC health — avoid deep discharge and overcharge
        if 0.2 <= soc <= 0.8:
            reward += 0.5 * soc
        else:
            reward -= 1.0

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        """
        Build the 12-dimensional normalised observation vector
        for the current timestep.
        """
        t       = min(self.t, self.T - 1)
        p_load  = float(self.p_load_series[t])
        p_solar = float(self.p_solar_series[t])
        soc     = self.battery.soc

        # Rolling statistics from in-episode history
        h_load  = self._load_history[-6:]  if self._load_history  else [p_load]
        h_solar = self._solar_history[-6:] if self._solar_history else [p_solar]
        h_soc   = self._soc_history[-1:]   if self._soc_history   else [soc]

        load_avg6  = float(np.mean(h_load))
        solar_avg6 = float(np.mean(h_solar))
        soc_roc    = float(soc - h_soc[-1])

        # Quick load flow for voltage / line loading in obs
        lf = run_load_flow(self.net, p_load, p_solar)

        load_lag1 = (
            float(self._load_history[-1]) / self.max_load
            if self._load_history else p_load / self.max_load
        )

        obs = np.array([
            soc,                                             # 0
            p_load  / self.max_load,                         # 1
            p_solar / self.max_solar,                        # 2
            lf['v_min'],                                     # 3
            (t % 24) / 23.0,                                 # 4
            load_lag1,                                       # 5
            solar_avg6 / self.max_solar,                     # 6
            load_avg6  / self.max_load,                      # 7
            (p_load / self.max_load)                         # 8 — grid stress
            + (1.0 - p_solar / self.max_solar)
            + (1.0 - soc),
            soc_roc,                                         # 9
            (p_load - p_solar) / self.max_load,              # 10
            lf['line_loading_max'] / 100.0,                  # 11
        ], dtype=np.float32)

        return obs


# ── Quick validation ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    from gymnasium.utils.env_checker import check_env

    LOAD_PATH  = r'datasets\LD2011_2014.txt'
    SOLAR_PATH = r'datasets\Plant_1_Generation_Data.csv'

    if not os.path.exists(LOAD_PATH) or not os.path.exists(SOLAR_PATH):
        print("[ERROR] Dataset files not found. Adjust paths and retry.")
    else:
        print("Creating MicrogridEnv ...")
        env = MicrogridEnv(LOAD_PATH, SOLAR_PATH, n_days=7)
        print(f"  Timesteps   : {env.T}")
        print(f"  Obs  space  : {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        print("\nRunning Gymnasium env checker ...")
        check_env(env, warn=True)
        print("[OK] Environment passed gym checks.\n")

        print("Running 1 random episode ...")
        obs, info = env.reset()
        total_reward = 0.0
        blackouts = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            blackouts    += info['blackout']

        print(f"Episode done.")
        print(f"  Total reward : {total_reward:.2f}")
        print(f"  Blackouts    : {blackouts} / {env.T}")
        print(f"  LOLP         : {blackouts / env.T * 100:.2f}%")
        print("[OK] rl_environment.py validated successfully.")
