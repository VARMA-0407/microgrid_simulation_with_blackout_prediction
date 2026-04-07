"""
Loads and preprocesses Plant 1 Generation Data.
Outputs hourly solar generation, normalized and scaled.
"""

import numpy as np
import pandas as pd

# IEEE 33-bus base solar capacity default
IEEE33_BASE_SOLAR_MW = 2.0


def load_solar_generation(path: str) -> pd.Series:
    """
    Read Plant 1 Generation dataset.

    - Aggregates AC_POWER across all inverters (SOURCE_KEYs).
    - Resamples from 15-min to hourly resolution.

    Parameters
    ----------
    path : str
        Path to Plant_1_Generation_Data.csv

    Returns
    -------
    pd.Series
        Hourly total AC power generation in MW with DatetimeIndex
    """
    df = pd.read_csv(path)

    # Parse dates (Plant 1 dataset usually uses dd-mm-yyyy HH:MM)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format="%d-%m-%Y %H:%M")
    df = df.set_index('DATE_TIME')

    # Sort in chronological order
    df = df.sort_index()

    # Group by index (DATE_TIME) and sum the AC_POWER across all inverters
    total_kw = df.groupby(level=0)['AC_POWER'].sum()

    # Fill gaps: forward-fill first, then backward-fill
    total_kw = total_kw.ffill().bfill()

    # Convert kW -> MW
    total_mw = total_kw / 1000.0

    # Downsample 15-min -> hourly by taking the mean power per hour
    hourly_mw = total_mw.resample('h').mean()

    # Drop any boundary NaNs
    hourly_mw = hourly_mw.dropna()

    return hourly_mw


def normalize_by_max(series: pd.Series) -> pd.Series:
    """
    Max-scale series to [0, 1].
    Preserves the shape of the generation profile.

    Parameters
    ----------
    series : pd.Series
        Raw generation in MW

    Returns
    -------
    pd.Series
        Normalized generation in range [0, 1]
    """
    max_val = series.max()
    if max_val == 0:
        return series
    return series / max_val


def scale_to_ieee_base(solar_norm: pd.Series, base_mw: float = IEEE33_BASE_SOLAR_MW) -> pd.Series:
    """
    Scale normalized [0, 1] solar generation to IEEE 33-bus feeder capacity.

    Parameters
    ----------
    solar_norm : pd.Series
        Normalized generation in [0, 1]
    base_mw : float
        IEEE 33-bus base solar capacity in MW (default 2.0)

    Returns
    -------
    pd.Series
        Solar generation scaled to MW range [0, base_mw]
    """
    return solar_norm * base_mw


def prepare_solar_series(path: str, n_days: int = 30) -> pd.Series:
    """
    Full pipeline: raw CSV -> simulation-ready hourly generation in MW.

    Steps:
        1. Load and clean raw solar data
        2. Trim to simulation horizon
        3. Normalize to [0, 1]
        4. Scale to IEEE 33-bus base capacity

    Parameters
    ----------
    path   : str
        Path to Plant_1_Generation_Data.csv
    n_days : int
        Simulation horizon in days (default 30)

    Returns
    -------
    pd.Series
        Integer-indexed hourly solar generation in MW, length = n_days * 24
    """
    raw    = load_solar_generation(path)
    trimmed = raw.iloc[: n_days * 24]
    norm   = normalize_by_max(trimmed)
    scaled = scale_to_ieee_base(norm)

    return scaled.reset_index(drop=True)


def validate_solar_series(series: pd.Series) -> None:
    """
    Sanity checks on the prepared solar series.
    Prints a summary if all checks pass.
    Raises AssertionError with a clear message if any check fails.

    Parameters
    ----------
    series : pd.Series
        Output of prepare_solar_series()
    """
    assert series.isna().sum() == 0, \
        f"Solar series contains {series.isna().sum()} NaN values."
    assert series.min() >= 0.0, \
        f"Solar series contains negative values (min={series.min():.4f})."
    assert series.max() <= IEEE33_BASE_SOLAR_MW + 1e-6, \
        f"Solar generation exceeds capacity: max={series.max():.4f} MW."

    print(
        f"[VALID] Solar series OK\n"
        f"        Length : {len(series)} hours\n"
        f"        Min    : {series.min():.4f} MW\n"
        f"        Max    : {series.max():.4f} MW\n"
        f"        Mean   : {series.mean():.4f} MW"
    )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    path   = sys.argv[1] if len(sys.argv) > 1 else r'datasets\Plant_1_Generation_Data.csv'
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"Loading data from: {path}")
    solar = prepare_solar_series(path, n_days=n_days)

    validate_solar_series(solar)

    print(f"\nFirst 5 values:\n{solar.head()}")