"""
Battery Model for Digital Twin Microgrid.

Rule-based battery with SOC (State of Charge) tracking.
Charges when solar surplus, discharges when load > solar.
"""

import numpy as np


class BatteryModel:
    """
    Rule-based battery energy storage system (BESS).

    At each timestep, decides whether to charge or discharge
    based on the balance between load and solar generation.

    Parameters
    ----------
    capacity_mwh : float
        Total energy capacity of the battery in MWh.
    soc_init : float
        Initial state of charge as a fraction (0.0 to 1.0).
    soc_min : float
        Minimum allowed SOC (protects against deep discharge).
    soc_max : float
        Maximum allowed SOC (protects against overcharge).
    charge_rate : float
        Maximum charging power in MW.
    discharge_rate : float
        Maximum discharging power in MW.
    eta : float
        Round-trip efficiency (0.0 to 1.0). Energy is lost
        in both charging and discharging directions.
    """

    def __init__(
        self,
        capacity_mwh: float = 2.0,
        soc_init: float = 0.5,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        charge_rate: float = 0.5,
        discharge_rate: float = 0.5,
        eta: float = 0.95,
    ):
        self.capacity = capacity_mwh
        self.soc = soc_init
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.eta = eta

    def step(self, p_load: float, p_solar: float) -> tuple:
        """
        Execute one timestep of battery dispatch.

        Logic:
            surplus > 0  →  solar exceeds load  →  CHARGE battery
            surplus <= 0 →  load exceeds solar   →  DISCHARGE battery

        Three constraints limit charge/discharge at each step:
            1. charge_rate / discharge_rate  — inverter power limit
            2. headroom_mwh                 — can't charge beyond soc_max
            3. available_mwh                — can't discharge below soc_min

        Parameters
        ----------
        p_load : float
            Load demand at this timestep (MW).
        p_solar : float
            Solar generation at this timestep (MW).

        Returns
        -------
        tuple(float, float)
            (p_battery, new_soc)
            p_battery > 0  →  discharging (supplying load)
            p_battery < 0  →  charging (absorbing surplus)
        """
        surplus = p_solar - p_load  # positive = excess solar

        if surplus > 0:
            # --- CHARGE ---
            headroom_mwh = (self.soc_max - self.soc) * self.capacity
            p_charge = min(surplus, self.charge_rate, headroom_mwh)
            self.soc += (p_charge * self.eta) / self.capacity
            p_battery = -p_charge  # negative = charging
        else:
            # --- DISCHARGE ---
            deficit = -surplus
            available_mwh = (self.soc - self.soc_min) * self.capacity
            p_discharge = min(deficit, self.discharge_rate, available_mwh)
            self.soc -= p_discharge / (self.capacity * self.eta)
            p_battery = p_discharge  # positive = discharging

        # Clamp SOC to valid range
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)

        return p_battery, self.soc

    def reset(self, soc_init: float = 0.5):
        """Reset SOC to initial value."""
        self.soc = soc_init

    def is_empty(self) -> bool:
        """Check if battery is at minimum SOC."""
        return self.soc <= self.soc_min + 1e-6

    def is_full(self) -> bool:
        """Check if battery is at maximum SOC."""
        return self.soc >= self.soc_max - 1e-6

    def get_status(self) -> dict:
        """Return current battery status as a dictionary."""
        return {
            'capacity_mwh': self.capacity,
            'soc': self.soc,
            'soc_min': self.soc_min,
            'soc_max': self.soc_max,
            'charge_rate': self.charge_rate,
            'discharge_rate': self.discharge_rate,
            'eta': self.eta,
            'is_empty': self.is_empty(),
            'is_full': self.is_full(),
        }

    def __repr__(self) -> str:
        return (
            f"BatteryModel("
            f"capacity={self.capacity} MWh, "
            f"SOC={self.soc:.4f}, "
            f"range=[{self.soc_min}, {self.soc_max}], "
            f"charge_rate={self.charge_rate} MW, "
            f"discharge_rate={self.discharge_rate} MW, "
            f"eta={self.eta})"
        )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("Battery Model — Quick Test")
    print("=" * 50)

    battery = BatteryModel(capacity_mwh=2.0, soc_init=0.5)
    print(f"\n{battery}\n")

    # Synthetic 24-hour scenario
    # Night (0-5):  load=2.5, solar=0.0  → discharge
    # Day   (6-17): load=2.0, solar=3.0  → charge
    # Eve   (18-23):load=3.0, solar=0.0  → discharge
    test_data = []
    for h in range(24):
        if h < 6:
            p_load, p_solar = 2.5, 0.0
        elif h < 18:
            p_load, p_solar = 2.0, 3.0
        else:
            p_load, p_solar = 3.0, 0.0
        test_data.append((h, p_load, p_solar))

    print(f"{'Hour':>4} | {'p_load':>8} | {'p_solar':>8} | {'p_batt':>8} | {'p_grid':>8} | {'SOC':>7}")
    print("-" * 58)

    for h, p_load, p_solar in test_data:
        p_battery, soc = battery.step(p_load, p_solar)
        p_grid = p_load - p_solar - p_battery
        print(f"{h:4d} | {p_load:8.3f} | {p_solar:8.3f} | {p_battery:8.3f} | {p_grid:8.3f} | {soc:.4f}")

    print(f"\nFinal SOC : {battery.soc:.4f}")
    print(f"Is empty  : {battery.is_empty()}")
    print(f"Is full   : {battery.is_full()}")
    print("\n[OK] Battery model test complete.")
