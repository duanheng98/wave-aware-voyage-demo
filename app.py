# app.py

import base64
import os
import heapq
import json
import math
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import folium
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
from PIL import Image
from streamlit_folium import st_folium


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Wave-Aware Voyage Optimisation Demo",
    layout="wide",
)

DATA_PATH = "data/wave_sample.nc"
SLP_DATA_PATH = "data/slp_sample.nc"
SLP_CONTOUR_INTERVAL_HPA = 4.0
SLP_MAX_CONTOUR_POINTS_PER_LINE = 500

DATA_BOUNDS = {
    "min_lon": -30,
    "max_lon": 15,
    "min_lat": 45,
    "max_lat": 66,
}

PORTS = {
    # Netherlands / Belgium / Germany
    "Rotterdam": (51.95, 4.14),
    "Amsterdam": (52.38, 4.90),
    "Antwerp": (51.26, 4.40),
    "Hamburg": (53.54, 9.98),
    "Bremerhaven": (53.54, 8.58),
    "Wilhelmshaven": (53.53, 8.13),

    # UK / Ireland
    "London / Thames": (51.50, 0.00),
    "Dover": (51.13, 1.31),
    "Southampton": (50.90, -1.40),
    "Liverpool": (53.41, -3.00),
    "Dublin": (53.35, -6.26),
    "Cork": (51.90, -8.47),
    "Belfast": (54.60, -5.93),
    "Aberdeen": (57.15, -2.09),
    "Edinburgh / Leith": (55.98, -3.17),
    "Newcastle": (54.98, -1.61),

    # France / Channel / Bay of Biscay
    "Le Havre": (49.49, 0.10),
    "Cherbourg": (49.64, -1.62),
    "Brest": (48.39, -4.49),
    "Saint-Nazaire": (47.28, -2.20),
    "La Rochelle": (46.16, -1.15),

    # Scandinavia / North Sea / Baltic entrance
    "Gothenburg": (57.70, 11.97),
    "Copenhagen": (55.68, 12.57),
    "Oslo": (59.91, 10.75),
    "Aarhus": (56.16, 10.21),
    "Esbjerg": (55.47, 8.45),
    "Stavanger": (58.97, 5.73),
    "Bergen": (60.39, 5.32),
    "Trondheim": (63.43, 10.39),

    # Atlantic islands
    "Reykjavik": (64.15, -21.94),
    "Tórshavn": (62.01, -6.77),
}

PORTS_IN_DOMAIN = {
    name: (lat, lon)
    for name, (lat, lon) in PORTS.items()
    if DATA_BOUNDS["min_lat"] <= lat <= DATA_BOUNDS["max_lat"]
    and DATA_BOUNDS["min_lon"] <= lon <= DATA_BOUNDS["max_lon"]
}

ROUTE_COLORS = {
    "Distance-only": "blue",
    "Fuel-oriented": "green",
    "Safety-oriented": "orange",
    "Balanced": "purple",
    "ETA-oriented": "red",
}

# Fixed internal model coefficients.
# These are kept out of the sidebar because they are prototype calibration
# values rather than operational settings for the user.
# User-facing controls should stay close to operational assumptions:
# departure, arrival, date/time, route objective, nominal vessel speed,
# operational Hs limit, and an optional vessel weather sensitivity multiplier.
SAFETY_WAVE_WEIGHT = 1.0
SHORT_WAVE_PENALTY_WEIGHT = 5.0
ABOVE_HS_LIMIT_PENALTY_WEIGHT = 20.0
SAFETY_DIRECTION_WEIGHT = 0.4

DEFAULT_MIN_EFFECTIVE_SPEED_KNOTS = 5.0

FUEL_WAVE_WEIGHT_BASE = 0.08
FUEL_SHORT_WAVE_WEIGHT_BASE = 0.4
FUEL_DIRECTION_WEIGHT_BASE = 0.05
SPEED_LOSS_WAVE_WEIGHT_BASE = 0.05
SPEED_LOSS_DIRECTION_WEIGHT_BASE = 0.10

# Simplified vessel performance presets. These values make fuel and CO2
# outputs interpretable in tonnes, but they are still assumptions for a demo.
# Replace them with a vessel-specific speed-power curve and SFOC data for a
# production-grade model.
VESSEL_PROFILES = {
    "Small cargo vessel": {
        "reference_speed_knots": 12.0,
        "reference_power_kw": 4500.0,
        "sfoc_g_per_kwh": 185.0,
    },
    "Feeder container vessel": {
        "reference_speed_knots": 16.0,
        "reference_power_kw": 12000.0,
        "sfoc_g_per_kwh": 175.0,
    },
    "RoRo / ferry style vessel": {
        "reference_speed_knots": 18.0,
        "reference_power_kw": 18000.0,
        "sfoc_g_per_kwh": 180.0,
    },
}

# Tank-to-wake CO2 conversion factors in tonnes CO2 per tonne fuel.
CO2_EMISSION_FACTORS = {
    "HFO": 3.114,
    "LFO": 3.151,
    "MDO/MGO": 3.206,
}

# Keep vessel and fuel assumptions implicit so the UI stays focused on the
# route-planning question: how metocean conditions affect time, fuel, CO2,
# and safety trade-offs. These can be changed here if a different demo vessel
# or fuel assumption is desired.
DEFAULT_VESSEL_PROFILE = "Small cargo vessel"
DEFAULT_FUEL_TYPE = "MDO/MGO"

POWER_SPEED_EXPONENT = 3.0
MAX_WEATHER_POWER_FACTOR = 4.0
MAX_ROUTE_DETOUR_FACTOR_FOR_WARNING = 1.8

# Routing guardrail: avoid paths that skim across land/no-data corners.
# This is a prototype alternative to official navigational constraints,
# coastline polygons, bathymetry, and traffic separation schemes.
NO_DATA_PROXIMITY_PENALTY_PER_NM = 200.0

# Time-dependent routing controls. These defaults keep the experimental
# time-aware A* search small enough for a laptop.
DEFAULT_TIME_BIN_MINUTES = 60
DEFAULT_CORRIDOR_WIDTH_NM = 150
DEFAULT_FORECAST_HORIZON_H = 72
MAX_ASTAR_EXPANSIONS = 300_000



# ============================================================
# Session state
# ============================================================

if "route_result" not in st.session_state:
    st.session_state.route_result = None


# ============================================================
# Utility functions
# ============================================================

@st.cache_resource
def load_dataset(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path)

    rename_map = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"

    if rename_map:
        ds = ds.rename(rename_map)

    if not np.all(np.diff(ds["latitude"].values) > 0):
        ds = ds.sortby("latitude")

    if not np.all(np.diff(ds["longitude"].values) > 0):
        ds = ds.sortby("longitude")

    return ds


@st.cache_resource
def load_slp_dataset(path: str) -> xr.Dataset:
    """Load optional sea-level pressure data and normalise common ERA5 names."""
    ds = xr.open_dataset(path)

    rename_map = {}
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename_map["valid_time"] = "time"
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"

    if rename_map:
        ds = ds.rename(rename_map)

    if "latitude" not in ds.coords or "longitude" not in ds.coords or "time" not in ds.coords:
        raise ValueError("SLP file must contain time, latitude, and longitude coordinates.")

    if not np.all(np.diff(ds["latitude"].values) > 0):
        ds = ds.sortby("latitude")

    lon_vals = np.asarray(ds["longitude"].values, dtype=float)
    if np.nanmax(lon_vals) > 180.0:
        wrapped_lons = ((lon_vals + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=wrapped_lons)

    if not np.all(np.diff(ds["longitude"].values) > 0):
        ds = ds.sortby("longitude")

    return ds


def detect_slp_variable(ds: xr.Dataset) -> str:
    """Return the most likely mean sea-level pressure variable name."""
    candidates = [
        "msl",
        "slp",
        "prmsl",
        "mean_sea_level_pressure",
        "mean_sea_level_pressure_at_mean_sea_level",
    ]
    for name in candidates:
        if name in ds.data_vars:
            return name

    pressure_like = []
    for name, da in ds.data_vars.items():
        units = str(da.attrs.get("units", "")).lower()
        long_name = str(da.attrs.get("long_name", "")).lower()
        standard_name = str(da.attrs.get("standard_name", "")).lower()
        text = " ".join([name.lower(), units, long_name, standard_name])
        if "sea" in text and "pressure" in text:
            pressure_like.append(name)
        elif name.lower() in {"msl", "slp"}:
            pressure_like.append(name)

    if pressure_like:
        return pressure_like[0]

    raise ValueError(
        "Could not detect an SLP variable. Expected something like 'msl' or 'slp'."
    )


def slp_to_hpa(values: np.ndarray, units: str = "") -> np.ndarray:
    """Convert a sea-level pressure field to hPa if it looks like Pa."""
    arr = np.asarray(values, dtype=float)
    units_lower = units.lower()
    if "pa" in units_lower and "hpa" not in units_lower:
        return arr / 100.0

    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmedian(finite) > 2000:
        return arr / 100.0

    return arr


def contour_lines_from_field(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    interval_hpa: float = SLP_CONTOUR_INTERVAL_HPA,
) -> List[Dict]:
    """Convert an SLP field into contour polylines for Leaflet."""
    arr = np.asarray(field, dtype=float)
    finite = np.isfinite(arr)
    if arr.ndim != 2 or not finite.any():
        return []

    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        return []

    start = math.ceil(min_val / interval_hpa) * interval_hpa
    stop = math.floor(max_val / interval_hpa) * interval_hpa
    if stop < start:
        return []

    levels = np.arange(start, stop + 0.5 * interval_hpa, interval_hpa)
    if levels.size == 0:
        return []

    fig = plt.figure(figsize=(1, 1))
    try:
        cs = plt.contour(lons, lats, arr, levels=levels)
        lines = []
        for level, segments in zip(cs.levels, cs.allsegs):
            for verts in segments:
                verts = np.asarray(verts, dtype=float)
                if verts.shape[0] < 2:
                    continue

                if verts.shape[0] > SLP_MAX_CONTOUR_POINTS_PER_LINE:
                    step = int(math.ceil(verts.shape[0] / SLP_MAX_CONTOUR_POINTS_PER_LINE))
                    verts = verts[::step]

                coords = [[float(lat), float(lon)] for lon, lat in verts]
                lines.append({"value": float(level), "coords": coords})
        return lines
    finally:
        plt.close(fig)


def slp_contours_for_time(
    slp_ds: Optional[xr.Dataset],
    target_time: pd.Timestamp,
    map_bounds: Dict[str, float],
) -> Dict:
    """Sample SLP near the selected time and return pressure contours."""
    if slp_ds is None:
        return {"available": False, "label": None, "lines": []}

    var_name = detect_slp_variable(slp_ds)
    snap = slp_ds.sel(time=np.datetime64(target_time), method="nearest")
    da = snap[var_name].squeeze(drop=True)

    if "latitude" not in da.coords or "longitude" not in da.coords:
        return {"available": False, "label": None, "lines": []}

    lat_min = max(float(np.nanmin(slp_ds["latitude"].values)), map_bounds["min_lat"])
    lat_max = min(float(np.nanmax(slp_ds["latitude"].values)), map_bounds["max_lat"])
    lon_min = max(float(np.nanmin(slp_ds["longitude"].values)), map_bounds["min_lon"])
    lon_max = min(float(np.nanmax(slp_ds["longitude"].values)), map_bounds["max_lon"])

    if lat_min >= lat_max or lon_min >= lon_max:
        return {"available": False, "label": None, "lines": []}

    da = da.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    if da.size == 0:
        return {"available": False, "label": None, "lines": []}

    values_hpa = slp_to_hpa(da.values, units=str(slp_ds[var_name].attrs.get("units", "")))
    lines = contour_lines_from_field(
        field=values_hpa,
        lats=np.asarray(da["latitude"].values, dtype=float),
        lons=np.asarray(da["longitude"].values, dtype=float),
    )

    slp_time = pd.Timestamp(snap["time"].values)
    return {
        "available": bool(lines),
        "label": slp_time.strftime("%Y-%m-%d %H:%M"),
        "lines": lines,
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    )

    brng = math.degrees(math.atan2(x, y))
    return (brng + 360) % 360


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest angular difference between two directions."""
    return abs((a - b + 180) % 360 - 180)


def classify_wave_relative_to_heading(wave_from_deg: float, heading_deg: float) -> str:
    """
    VMDR is usually wave-from direction.
    If wave_from direction is close to vessel heading, the ship is facing head seas.
    """
    rel = angular_difference_deg(wave_from_deg, heading_deg)

    if rel <= 45:
        return "head"
    elif rel < 135:
        return "beam"
    else:
        return "following"


def directional_penalty(wave_from_deg: float, heading_deg: float) -> float:
    """
    Simple operational penalty:
    - head sea: strongest speed loss / fuel effect
    - beam sea: rolling / comfort / cargo risk
    - following sea: lower fuel penalty but not risk-free
    """
    category = classify_wave_relative_to_heading(wave_from_deg, heading_deg)

    if category == "head":
        return 1.0
    elif category == "beam":
        return 0.7
    else:
        return 0.3


def nearest_ocean_cell(
    lat: float,
    lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[int, int]:
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    dist2 = (lat_grid - lat) ** 2 + (lon_grid - lon) ** 2
    dist2 = np.where(valid_mask, dist2, np.inf)

    if not np.isfinite(dist2).any():
        raise ValueError("No valid ocean cells found in this domain.")

    i, j = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(i), int(j)


def edge_stays_in_valid_water(
    i: int,
    j: int,
    ni: int,
    nj: int,
    valid_mask: np.ndarray,
) -> bool:
    """Return False for grid edges that cut through no-data/land corners.

    Both endpoints can be valid ocean cells while a diagonal move still slices
    across a land/no-data corner. For diagonal moves, require the two adjacent
    orthogonal cells to be valid too. This keeps the route inside connected
    water cells instead of visually crossing transparent raster patches.
    """
    if not (valid_mask[i, j] and valid_mask[ni, nj]):
        return False

    di = ni - i
    dj = nj - j

    if abs(di) == 1 and abs(dj) == 1:
        return bool(valid_mask[i, nj] and valid_mask[ni, j])

    return True


def no_data_proximity_penalty(valid_mask: np.ndarray) -> np.ndarray:
    """Penalty field for valid cells next to land/no-data cells.

    The value is 0 for interior ocean cells and approaches 1 near no-data
    boundaries. It discourages routes from hugging the edge of the available
    wave field, while still allowing coastal port access when necessary.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    nlat, nlon = mask.shape
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)

    valid_neighbour_count = np.zeros((nlat, nlon), dtype=float)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            valid_neighbour_count += padded[1 + di:1 + di + nlat, 1 + dj:1 + dj + nlon]

    invalid_fraction = 1.0 - valid_neighbour_count / 8.0
    return np.where(mask, invalid_fraction, np.inf)


def path_to_latlon(
    path: List[Tuple[int, int]],
    lats: np.ndarray,
    lons: np.ndarray,
) -> List[Tuple[float, float]]:
    return [(float(lats[i]), float(lons[j])) for i, j in path]


def path_distance_km(latlon_path: List[Tuple[float, float]]) -> float:
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(latlon_path[:-1], latlon_path[1:]):
        total += haversine_km(lat1, lon1, lat2, lon2)
    return total


def compute_cell_fields(
    hs: np.ndarray,
    tp: np.ndarray,
    wave_limit: float,
    safety_wave_weight: float,
    short_wave_penalty_weight: float,
    above_hs_limit_penalty_weight: float,
    fuel_wave_weight: float,
    fuel_steepness_weight: float,
) -> Dict[str, np.ndarray]:
    """
    Split the problem into separate components:

    1. Fuel factor:
       Mild sea-state-related resistance proxy.
       Does NOT include hard safety threshold.

    2. Safety risk:
       Higher penalty for high waves, short-period seas, and Hs limit exceedance.

    3. Short-wave proxy:
       Hs / Tp, simple proxy for short-period, steep-feeling seas.
    """
    tp_safe = np.maximum(tp, 0.1)
    steepness = hs / tp_safe

    fuel_factor = (
        1.0
        + fuel_wave_weight * hs**2
        + fuel_steepness_weight * steepness
    )

    safety_risk = (
        safety_wave_weight * hs**2
        + short_wave_penalty_weight * steepness
        + above_hs_limit_penalty_weight * np.maximum(0, hs - wave_limit) ** 2
    )

    return {
        "steepness": steepness,
        "fuel_factor": fuel_factor,
        "safety_risk": safety_risk,
    }


def calm_power_kw(
    speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    exponent: float = POWER_SPEED_EXPONENT,
) -> float:
    """Simple speed-power relation used for the demo fuel model."""
    if reference_speed_knots <= 0:
        raise ValueError("reference_speed_knots must be positive.")

    speed_ratio = max(speed_knots, 0.1) / reference_speed_knots
    return reference_power_kw * speed_ratio**exponent


def segment_components(
    i: int,
    j: int,
    ni: int,
    nj: int,
    lats: np.ndarray,
    lons: np.ndarray,
    hs: np.ndarray,
    tp: np.ndarray,
    vmdr: np.ndarray,
    fuel_factor: np.ndarray,
    safety_risk: np.ndarray,
    ship_speed_knots: float,
    min_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
) -> Dict[str, float]:
    """
    Compute segment-level voyage components.

    Fuel estimate:
      fuel_tonnes = engine_power_kw * SFOC_g_per_kWh * time_h / 1,000,000

    CO2 estimate:
      co2_tonnes = fuel_tonnes * fuel_specific_emission_factor

    Time estimate:
      distance_nm / effective_speed

    Safety proxy:
      distance_nm * local_safety_risk

    Direction penalty:
      Uses VMDR relative to vessel heading.
    """
    lat1, lon1 = float(lats[i]), float(lons[j])
    lat2, lon2 = float(lats[ni]), float(lons[nj])

    distance_km = haversine_km(lat1, lon1, lat2, lon2)
    distance_nm = distance_km / 1.852

    heading = bearing_deg(lat1, lon1, lat2, lon2)

    hs_local = float(0.5 * (hs[i, j] + hs[ni, nj]))
    tp_local = float(0.5 * (tp[i, j] + tp[ni, nj]))
    wave_from_local = float(vmdr[i, j])

    dir_penalty = directional_penalty(wave_from_local, heading)

    local_fuel_factor = float(0.5 * (fuel_factor[i, j] + fuel_factor[ni, nj]))
    local_safety_risk = float(0.5 * (safety_risk[i, j] + safety_risk[ni, nj]))

    # Direction affects added resistance. Head seas cost more than following seas.
    local_fuel_factor = local_fuel_factor * (
        1.0 + fuel_direction_weight * dir_penalty * hs_local
    )

    # Guardrail for the demo: very high sensitivity can otherwise create
    # unrealistically large added-power multipliers and encourage extreme
    # detours. A production model should replace this with a calibrated
    # vessel-specific speed-power / added-resistance model.
    local_fuel_factor = min(local_fuel_factor, MAX_WEATHER_POWER_FACTOR)

    # Direction affects safety/comfort too.
    local_safety_risk = local_safety_risk * (
        1.0 + safety_direction_weight * dir_penalty
    )

    # Simple speed-loss proxy.
    # Bad waves and head/beam sea reduce effective speed over the segment.
    speed_loss = (
        speed_loss_wave_weight * hs_local**2
        + speed_loss_direction_weight * dir_penalty * hs_local
    )

    effective_speed = max(min_speed_knots, ship_speed_knots - speed_loss)
    segment_time = distance_nm / effective_speed

    # Semi-realistic fuel estimate.
    # The baseline power comes from a simple speed-power curve. Waves then
    # multiply power demand through local_fuel_factor. This is not a substitute
    # for a vessel-specific speed-power table, but it gives tonnes fuel instead
    # of an arbitrary index.
    base_power_kw = calm_power_kw(
        speed_knots=ship_speed_knots,
        reference_speed_knots=reference_speed_knots,
        reference_power_kw=reference_power_kw,
    )
    weather_adjusted_power_kw = base_power_kw * local_fuel_factor
    segment_fuel_tonnes = (
        weather_adjusted_power_kw * sfoc_g_per_kwh * segment_time / 1_000_000
    )
    segment_co2_tonnes = segment_fuel_tonnes * co2_emission_factor
    segment_safety = distance_nm * local_safety_risk

    return {
        "distance_km": distance_km,
        "distance_nm": distance_nm,
        "fuel": segment_fuel_tonnes,
        "co2": segment_co2_tonnes,
        "time": segment_time,
        "safety": segment_safety,
        "effective_speed": effective_speed,
        "engine_power_kw": weather_adjusted_power_kw,
        "hs": hs_local,
        "tp": tp_local,
        "wave_from_deg": wave_from_local,
        "heading_deg": heading,
        "wave_relative_angle_deg": angular_difference_deg(wave_from_local, heading),
        "wave_relative_category": classify_wave_relative_to_heading(wave_from_local, heading),
        "direction_penalty": dir_penalty,
    }



def weighted_objective_cost(
    comp: Dict[str, float],
    objective_weights: Dict[str, float],
) -> float:
    """Combine segment components into the selected route objective."""
    return float(
        objective_weights.get("fuel", 0.0) * comp["fuel"]
        + objective_weights.get("time", 0.0) * comp["time"]
        + objective_weights.get("safety", 0.0) * comp["safety"]
        + objective_weights.get("distance", 0.0) * comp["distance_nm"]
    )


def local_segment_components(
    i: int,
    j: int,
    ni: int,
    nj: int,
    lats: np.ndarray,
    lons: np.ndarray,
    hs_local: float,
    tp_local: float,
    wave_from_local: float,
    wave_limit: float,
    safety_wave_weight: float,
    short_wave_penalty_weight: float,
    above_hs_limit_penalty_weight: float,
    fuel_wave_weight: float,
    fuel_steepness_weight: float,
    ship_speed_knots: float,
    min_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
) -> Dict[str, float]:
    """Segment calculation using already-sampled local weather values.

    This is used by the time-dependent A* search. Unlike the static
    segment_components(), it does not receive whole 2D wave fields. The search
    samples Hs/Tp/VMDR at the segment's estimated passage time, then this
    function turns those local values into distance, time, fuel, CO2 and safety.
    """
    lat1, lon1 = float(lats[i]), float(lons[j])
    lat2, lon2 = float(lats[ni]), float(lons[nj])

    distance_km = haversine_km(lat1, lon1, lat2, lon2)
    distance_nm = distance_km / 1.852
    heading = bearing_deg(lat1, lon1, lat2, lon2)

    tp_safe = max(float(tp_local), 0.1)
    short_wave_proxy = float(hs_local) / tp_safe

    local_fuel_factor = (
        1.0
        + fuel_wave_weight * float(hs_local) ** 2
        + fuel_steepness_weight * short_wave_proxy
    )

    local_safety_risk = (
        safety_wave_weight * float(hs_local) ** 2
        + short_wave_penalty_weight * short_wave_proxy
        + above_hs_limit_penalty_weight * max(0.0, float(hs_local) - wave_limit) ** 2
    )

    dir_penalty = directional_penalty(float(wave_from_local), heading)

    local_fuel_factor *= 1.0 + fuel_direction_weight * dir_penalty * float(hs_local)
    local_fuel_factor = min(local_fuel_factor, MAX_WEATHER_POWER_FACTOR)

    local_safety_risk *= 1.0 + safety_direction_weight * dir_penalty

    speed_loss = (
        speed_loss_wave_weight * float(hs_local) ** 2
        + speed_loss_direction_weight * dir_penalty * float(hs_local)
    )
    effective_speed = max(min_speed_knots, ship_speed_knots - speed_loss)
    segment_time = distance_nm / effective_speed

    base_power_kw = calm_power_kw(
        speed_knots=ship_speed_knots,
        reference_speed_knots=reference_speed_knots,
        reference_power_kw=reference_power_kw,
    )
    weather_adjusted_power_kw = base_power_kw * local_fuel_factor
    segment_fuel_tonnes = (
        weather_adjusted_power_kw * sfoc_g_per_kwh * segment_time / 1_000_000
    )
    segment_co2_tonnes = segment_fuel_tonnes * co2_emission_factor
    segment_safety = distance_nm * local_safety_risk

    return {
        "distance_km": distance_km,
        "distance_nm": distance_nm,
        "fuel": segment_fuel_tonnes,
        "co2": segment_co2_tonnes,
        "time": segment_time,
        "safety": segment_safety,
        "effective_speed": effective_speed,
        "engine_power_kw": weather_adjusted_power_kw,
        "hs": float(hs_local),
        "tp": float(tp_local),
        "wave_from_deg": float(wave_from_local),
        "heading_deg": heading,
        "wave_relative_angle_deg": angular_difference_deg(float(wave_from_local), heading),
        "wave_relative_category": classify_wave_relative_to_heading(float(wave_from_local), heading),
        "short_wave_proxy": short_wave_proxy,
        "direction_penalty": dir_penalty,
    }


def time_interpolation_weights(
    time_values_ns: np.ndarray,
    target_time: pd.Timestamp,
) -> Optional[Tuple[int, int, float]]:
    """Return bracketing time indices and linear interpolation weight."""
    if len(time_values_ns) == 0:
        return None

    target_ns = np.datetime64(target_time, "ns").astype("int64")

    if target_ns < time_values_ns[0] or target_ns > time_values_ns[-1]:
        return None

    idx = int(np.searchsorted(time_values_ns, target_ns, side="left"))

    if idx == 0:
        return 0, 0, 0.0

    if idx < len(time_values_ns) and time_values_ns[idx] == target_ns:
        return idx, idx, 0.0

    if idx >= len(time_values_ns):
        return len(time_values_ns) - 1, len(time_values_ns) - 1, 0.0

    i0 = idx - 1
    i1 = idx
    dt = float(time_values_ns[i1] - time_values_ns[i0])
    if dt <= 0:
        return i0, i0, 0.0

    alpha = float(target_ns - time_values_ns[i0]) / dt
    return i0, i1, alpha


def sample_edge_wave_at_time(
    i: int,
    j: int,
    ni: int,
    nj: int,
    departure_time: pd.Timestamp,
    hours_from_departure: float,
    time_values_ns: np.ndarray,
    hs_cube: np.ndarray,
    tp_cube: np.ndarray,
    vmdr_sin_cube: np.ndarray,
    vmdr_cos_cube: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Sample Hs, Tp and VMDR for an edge at a voyage-relative time.

    Hs and Tp are linearly interpolated in time. VMDR is interpolated through
    sine/cosine components so 359° and 1° do not average to 180°.
    """
    target_time = departure_time + pd.Timedelta(hours=float(hours_from_departure))
    weights = time_interpolation_weights(time_values_ns, target_time)
    if weights is None:
        return None

    t0, t1, alpha = weights

    def edge_mean(cube: np.ndarray, t: int) -> float:
        return float(0.5 * (cube[t, i, j] + cube[t, ni, nj]))

    hs0 = edge_mean(hs_cube, t0)
    tp0 = edge_mean(tp_cube, t0)
    sin0 = edge_mean(vmdr_sin_cube, t0)
    cos0 = edge_mean(vmdr_cos_cube, t0)

    if t0 == t1:
        hs_val, tp_val = hs0, tp0
        sin_val, cos_val = sin0, cos0
    else:
        hs1 = edge_mean(hs_cube, t1)
        tp1 = edge_mean(tp_cube, t1)
        sin1 = edge_mean(vmdr_sin_cube, t1)
        cos1 = edge_mean(vmdr_cos_cube, t1)

        hs_val = (1.0 - alpha) * hs0 + alpha * hs1
        tp_val = (1.0 - alpha) * tp0 + alpha * tp1
        sin_val = (1.0 - alpha) * sin0 + alpha * sin1
        cos_val = (1.0 - alpha) * cos0 + alpha * cos1

    if not all(np.isfinite(v) for v in (hs_val, tp_val, sin_val, cos_val)):
        return None

    wave_from = (math.degrees(math.atan2(sin_val, cos_val)) + 360.0) % 360.0
    return float(hs_val), float(tp_val), float(wave_from)


def dilate_mask_by_cells(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Pure NumPy circular dilation for route corridor masks."""
    if radius_cells <= 0:
        return np.asarray(mask, dtype=bool).copy()

    base = np.asarray(mask, dtype=bool)
    nlat, nlon = base.shape
    out = np.zeros_like(base, dtype=bool)

    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            if di * di + dj * dj > radius_cells * radius_cells:
                continue

            dst_i0 = max(0, di)
            dst_i1 = min(nlat, nlat + di)
            src_i0 = max(0, -di)
            src_i1 = min(nlat, nlat - di)

            dst_j0 = max(0, dj)
            dst_j1 = min(nlon, nlon + dj)
            src_j0 = max(0, -dj)
            src_j1 = min(nlon, nlon - dj)

            if dst_i0 < dst_i1 and dst_j0 < dst_j1:
                out[dst_i0:dst_i1, dst_j0:dst_j1] |= base[src_i0:src_i1, src_j0:src_j1]

    return out


def approximate_grid_spacing_nm(lats: np.ndarray, lons: np.ndarray) -> float:
    """Approximate one grid step in nautical miles near the domain centre."""
    lat_step = float(np.nanmedian(np.abs(np.diff(lats)))) if len(lats) > 1 else 0.1
    lon_step = float(np.nanmedian(np.abs(np.diff(lons)))) if len(lons) > 1 else 0.1
    mid_lat = float(np.nanmean(lats))

    lat_nm = max(lat_step * 60.0, 0.1)
    lon_nm = max(lon_step * 60.0 * math.cos(math.radians(mid_lat)), 0.1)
    return max(min(lat_nm, lon_nm), 0.1)


def build_route_corridor_mask(
    baseline_path: List[Tuple[int, int]],
    valid_mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    corridor_width_nm: float,
) -> np.ndarray:
    """Create a corridor around a baseline route using grid-cell dilation."""
    path_mask = np.zeros_like(valid_mask, dtype=bool)
    for i, j in baseline_path:
        if 0 <= i < path_mask.shape[0] and 0 <= j < path_mask.shape[1]:
            path_mask[i, j] = True

    grid_nm = approximate_grid_spacing_nm(lats, lons)
    radius_cells = max(1, int(math.ceil(corridor_width_nm / grid_nm)))
    return dilate_mask_by_cells(path_mask, radius_cells) & valid_mask


def objective_heuristic_to_target(
    i: int,
    j: int,
    end_node: Tuple[int, int],
    lats: np.ndarray,
    lons: np.ndarray,
    objective_weights: Dict[str, float],
    ship_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
) -> float:
    """Admissible-ish lower-bound heuristic for A*.

    It uses straight-line remaining distance, nominal speed, and calm-water
    fuel. Safety and no-data penalties are left at zero, which keeps the
    heuristic optimistic.
    """
    ei, ej = end_node
    remaining_nm = haversine_km(
        float(lats[i]), float(lons[j]), float(lats[ei]), float(lons[ej])
    ) / 1.852

    nominal_speed = max(ship_speed_knots, 0.1)
    min_remaining_time_h = remaining_nm / nominal_speed

    base_power_kw = calm_power_kw(
        speed_knots=ship_speed_knots,
        reference_speed_knots=reference_speed_knots,
        reference_power_kw=reference_power_kw,
    )
    calm_fuel_per_nm = base_power_kw * sfoc_g_per_kwh / 1_000_000 / nominal_speed

    return float(
        objective_weights.get("distance", 0.0) * remaining_nm
        + objective_weights.get("time", 0.0) * min_remaining_time_h
        + objective_weights.get("fuel", 0.0) * calm_fuel_per_nm * remaining_nm
    )


def reconstruct_state_path(
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]],
    final_state: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    path = [final_state]
    current = final_state
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def time_dependent_a_star_route(
    start_node: Tuple[int, int],
    end_node: Tuple[int, int],
    lats: np.ndarray,
    lons: np.ndarray,
    valid_mask: np.ndarray,
    search_mask: np.ndarray,
    no_data_penalty: np.ndarray,
    objective_weights: Dict[str, float],
    departure_time: pd.Timestamp,
    time_values_ns: np.ndarray,
    hs_cube: np.ndarray,
    tp_cube: np.ndarray,
    vmdr_sin_cube: np.ndarray,
    vmdr_cos_cube: np.ndarray,
    max_horizon_h: float,
    time_bin_minutes: int,
    wave_limit: float,
    safety_wave_weight: float,
    short_wave_penalty_weight: float,
    above_hs_limit_penalty_weight: float,
    fuel_wave_weight: float,
    fuel_steepness_weight: float,
    ship_speed_knots: float,
    min_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """Lazy time-dependent A* route search.

    State = (i, j, time_bin). The real arrival time is stored separately and
    is monotonically increased by each segment's computed travel time. The
    wave field is sampled at the current state's arrival time when expanding
    the next segment.
    """
    neighbours = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    nlat, nlon = valid_mask.shape
    bin_h = max(float(time_bin_minutes) / 60.0, 1.0 / 60.0)
    max_bin = int(math.ceil(max_horizon_h / bin_h))

    source = (start_node[0], start_node[1], 0)
    g_score: Dict[Tuple[int, int, int], float] = {source: 0.0}
    arrival_h: Dict[Tuple[int, int, int], float] = {source: 0.0}
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

    counter = 0
    first_priority = objective_heuristic_to_target(
        start_node[0], start_node[1], end_node, lats, lons,
        objective_weights, ship_speed_knots, reference_speed_knots,
        reference_power_kw, sfoc_g_per_kwh,
    )
    queue = [(first_priority, counter, source)]
    expanded = 0

    while queue:
        _, _, state = heapq.heappop(queue)
        current_cost = g_score.get(state, np.inf)
        current_time_h = arrival_h.get(state, np.inf)
        i, j, k = state

        if not np.isfinite(current_cost) or not np.isfinite(current_time_h):
            continue

        if (i, j) == end_node:
            state_path = reconstruct_state_path(came_from, state)
            path = [(si, sj) for si, sj, _ in state_path]
            return path, {
                "astar_expanded_states": float(expanded),
                "astar_final_time_h": float(current_time_h),
                "astar_final_cost": float(current_cost),
                "astar_time_bins": float(len({sk for _, _, sk in state_path})),
            }

        expanded += 1
        if expanded > MAX_ASTAR_EXPANSIONS:
            raise RuntimeError(
                f"A* exceeded {MAX_ASTAR_EXPANSIONS:,} expanded states. "
                "Try a wider time bin, a narrower corridor, or static routing."
            )

        if current_time_h > max_horizon_h or k > max_bin:
            continue

        for di, dj in neighbours:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= nlat or nj < 0 or nj >= nlon:
                continue
            if not search_mask[ni, nj]:
                continue
            if not edge_stays_in_valid_water(i, j, ni, nj, valid_mask):
                continue

            sampled = sample_edge_wave_at_time(
                i=i,
                j=j,
                ni=ni,
                nj=nj,
                departure_time=departure_time,
                hours_from_departure=current_time_h,
                time_values_ns=time_values_ns,
                hs_cube=hs_cube,
                tp_cube=tp_cube,
                vmdr_sin_cube=vmdr_sin_cube,
                vmdr_cos_cube=vmdr_cos_cube,
            )
            if sampled is None:
                continue

            hs_local, tp_local, wave_from_local = sampled
            comp = local_segment_components(
                i=i,
                j=j,
                ni=ni,
                nj=nj,
                lats=lats,
                lons=lons,
                hs_local=hs_local,
                tp_local=tp_local,
                wave_from_local=wave_from_local,
                wave_limit=wave_limit,
                safety_wave_weight=safety_wave_weight,
                short_wave_penalty_weight=short_wave_penalty_weight,
                above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
                fuel_wave_weight=fuel_wave_weight,
                fuel_steepness_weight=fuel_steepness_weight,
                ship_speed_knots=ship_speed_knots,
                min_speed_knots=min_speed_knots,
                reference_speed_knots=reference_speed_knots,
                reference_power_kw=reference_power_kw,
                sfoc_g_per_kwh=sfoc_g_per_kwh,
                co2_emission_factor=co2_emission_factor,
                speed_loss_wave_weight=speed_loss_wave_weight,
                speed_loss_direction_weight=speed_loss_direction_weight,
                fuel_direction_weight=fuel_direction_weight,
                safety_direction_weight=safety_direction_weight,
            )

            segment_weight = weighted_objective_cost(comp, objective_weights)
            local_no_data_penalty = float(
                0.5 * (no_data_penalty[i, j] + no_data_penalty[ni, nj])
                * comp["distance_nm"]
            )
            segment_weight += NO_DATA_PROXIMITY_PENALTY_PER_NM * local_no_data_penalty

            next_time_h = current_time_h + comp["time"]
            if next_time_h > max_horizon_h:
                continue

            next_bin = int(math.ceil(next_time_h / bin_h - 1e-9))
            next_bin = min(max(next_bin, k), max_bin)
            next_state = (ni, nj, next_bin)
            tentative_cost = current_cost + segment_weight

            if tentative_cost + 1e-9 < g_score.get(next_state, np.inf):
                g_score[next_state] = tentative_cost
                arrival_h[next_state] = next_time_h
                came_from[next_state] = state

                heuristic = objective_heuristic_to_target(
                    ni, nj, end_node, lats, lons, objective_weights,
                    ship_speed_knots, reference_speed_knots,
                    reference_power_kw, sfoc_g_per_kwh,
                )
                counter += 1
                heapq.heappush(queue, (tentative_cost + heuristic, counter, next_state))

    raise nx.NetworkXNoPath("No time-dependent route found within the selected horizon/corridor.")


def route_metrics_time_dependent(
    path: List[Tuple[int, int]],
    lats: np.ndarray,
    lons: np.ndarray,
    departure_time: pd.Timestamp,
    time_values_ns: np.ndarray,
    hs_cube: np.ndarray,
    tp_cube: np.ndarray,
    vmdr_sin_cube: np.ndarray,
    vmdr_cos_cube: np.ndarray,
    wave_limit: float,
    safety_wave_weight: float,
    short_wave_penalty_weight: float,
    above_hs_limit_penalty_weight: float,
    fuel_wave_weight: float,
    fuel_steepness_weight: float,
    ship_speed_knots: float,
    min_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
) -> Dict[str, float]:
    """Re-evaluate a route using continuous accumulated voyage time."""
    total_distance_km = 0.0
    total_distance_nm = 0.0
    total_fuel_tonnes = 0.0
    total_co2_tonnes = 0.0
    total_time = 0.0
    total_safety = 0.0
    total_power_time = 0.0

    hs_values, tp_values, short_wave_values, safety_values = [], [], [], []
    effective_speeds = []
    head_sea_segments = beam_sea_segments = following_sea_segments = 0

    for (i1, j1), (i2, j2) in zip(path[:-1], path[1:]):
        sampled = sample_edge_wave_at_time(
            i=i1,
            j=j1,
            ni=i2,
            nj=j2,
            departure_time=departure_time,
            hours_from_departure=total_time,
            time_values_ns=time_values_ns,
            hs_cube=hs_cube,
            tp_cube=tp_cube,
            vmdr_sin_cube=vmdr_sin_cube,
            vmdr_cos_cube=vmdr_cos_cube,
        )
        if sampled is None:
            raise RuntimeError("Route evaluation exceeded available wave data time range.")

        hs_local, tp_local, wave_from_local = sampled
        comp = local_segment_components(
            i=i1,
            j=j1,
            ni=i2,
            nj=j2,
            lats=lats,
            lons=lons,
            hs_local=hs_local,
            tp_local=tp_local,
            wave_from_local=wave_from_local,
            wave_limit=wave_limit,
            safety_wave_weight=safety_wave_weight,
            short_wave_penalty_weight=short_wave_penalty_weight,
            above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
            fuel_wave_weight=fuel_wave_weight,
            fuel_steepness_weight=fuel_steepness_weight,
            ship_speed_knots=ship_speed_knots,
            min_speed_knots=min_speed_knots,
            reference_speed_knots=reference_speed_knots,
            reference_power_kw=reference_power_kw,
            sfoc_g_per_kwh=sfoc_g_per_kwh,
            co2_emission_factor=co2_emission_factor,
            speed_loss_wave_weight=speed_loss_wave_weight,
            speed_loss_direction_weight=speed_loss_direction_weight,
            fuel_direction_weight=fuel_direction_weight,
            safety_direction_weight=safety_direction_weight,
        )

        total_distance_km += comp["distance_km"]
        total_distance_nm += comp["distance_nm"]
        total_fuel_tonnes += comp["fuel"]
        total_co2_tonnes += comp["co2"]
        total_safety += comp["safety"]
        total_power_time += comp["engine_power_kw"] * comp["time"]
        total_time += comp["time"]

        effective_speeds.append(comp["effective_speed"])
        hs_values.append(comp["hs"])
        tp_values.append(comp["tp"])
        short_wave_values.append(comp["short_wave_proxy"])
        safety_values.append(comp["safety"] / max(comp["distance_nm"], 1e-9))

        heading = bearing_deg(lats[i1], lons[j1], lats[i2], lons[j2])
        category = classify_wave_relative_to_heading(wave_from_local, heading)
        if category == "head":
            head_sea_segments += 1
        elif category == "beam":
            beam_sea_segments += 1
        else:
            following_sea_segments += 1

    nseg = max(1, len(path) - 1)
    mean_engine_power_kw = total_power_time / total_time if total_time > 0 else np.nan
    hs_arr = np.asarray(hs_values, dtype=float)
    tp_arr = np.asarray(tp_values, dtype=float)
    short_arr = np.asarray(short_wave_values, dtype=float)
    safety_arr = np.asarray(safety_values, dtype=float)

    return {
        "Distance (km)": total_distance_km,
        "Distance (nm)": total_distance_nm,
        "ETA / time proxy (h)": total_time,
        "Mean effective speed (knots)": float(np.nanmean(effective_speeds)) if effective_speeds else np.nan,
        "Estimated fuel consumption (t)": total_fuel_tonnes,
        "Estimated CO₂ emissions (t)": total_co2_tonnes,
        "Mean engine power (kW)": mean_engine_power_kw,
        "Safety risk index": total_safety,
        "Mean Hs (m)": float(np.nanmean(hs_arr)) if hs_arr.size else np.nan,
        "P90 Hs (m)": float(np.nanpercentile(hs_arr, 90)) if hs_arr.size else np.nan,
        "Max Hs (m)": float(np.nanmax(hs_arr)) if hs_arr.size else np.nan,
        f"Route cells Hs > {wave_limit:.1f}m (%)": float(np.mean(hs_arr > wave_limit) * 100) if hs_arr.size else np.nan,
        "Mean Tp (s)": float(np.nanmean(tp_arr)) if tp_arr.size else np.nan,
        "Mean weather power factor": np.nan,
        "Mean safety risk": float(np.nanmean(safety_arr)) if safety_arr.size else np.nan,
        "Head-sea segments (%)": 100 * head_sea_segments / nseg,
        "Beam-sea segments (%)": 100 * beam_sea_segments / nseg,
        "Following-sea segments (%)": 100 * following_sea_segments / nseg,
    }


def route_trajectory_static(
    path: List[Tuple[int, int]],
    lats: np.ndarray,
    lons: np.ndarray,
    hs: np.ndarray,
    tp: np.ndarray,
    vmdr: np.ndarray,
    fuel_factor: np.ndarray,
    safety_risk: np.ndarray,
    wave_limit: float,
    ship_speed_knots: float,
    min_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
) -> pd.DataFrame:
    """Build a cumulative time trajectory for a static-snapshot route."""
    if not path:
        return pd.DataFrame()

    rows = []
    total_time = 0.0
    i0, j0 = path[0]
    rows.append(
        {
            "cum_time_h": 0.0,
            "lat": float(lats[i0]),
            "lon": float(lons[j0]),
            "segment_index": 0,
            "hs_m": np.nan,
            "tp_s": np.nan,
            "wave_from_deg": np.nan,
            "heading_deg": np.nan,
            "wave_relative_angle_deg": np.nan,
            "effective_speed_knots": np.nan,
            "fuel_t": 0.0,
            "co2_t": 0.0,
            "safety": 0.0,
        }
    )

    for seg_idx, ((i1, j1), (i2, j2)) in enumerate(zip(path[:-1], path[1:]), start=1):
        comp = segment_components(
            i=i1,
            j=j1,
            ni=i2,
            nj=j2,
            lats=lats,
            lons=lons,
            hs=hs,
            tp=tp,
            vmdr=vmdr,
            fuel_factor=fuel_factor,
            safety_risk=safety_risk,
            ship_speed_knots=ship_speed_knots,
            min_speed_knots=min_speed_knots,
            reference_speed_knots=reference_speed_knots,
            reference_power_kw=reference_power_kw,
            sfoc_g_per_kwh=sfoc_g_per_kwh,
            co2_emission_factor=co2_emission_factor,
            speed_loss_wave_weight=speed_loss_wave_weight,
            speed_loss_direction_weight=speed_loss_direction_weight,
            fuel_direction_weight=fuel_direction_weight,
            safety_direction_weight=safety_direction_weight,
        )
        total_time += comp["time"]
        rows.append(
            {
                "cum_time_h": float(total_time),
                "lat": float(lats[i2]),
                "lon": float(lons[j2]),
                "segment_index": seg_idx,
                "hs_m": float(comp["hs"]),
                "tp_s": float(comp["tp"]),
                "wave_from_deg": float(comp.get("wave_from_deg", np.nan)),
                "heading_deg": float(comp.get("heading_deg", np.nan)),
                "wave_relative_angle_deg": float(comp.get("wave_relative_angle_deg", np.nan)),
                "effective_speed_knots": float(comp["effective_speed"]),
                "fuel_t": float(comp["fuel"]),
                "co2_t": float(comp["co2"]),
                "safety": float(comp["safety"]),
            }
        )

    return pd.DataFrame(rows)


def route_trajectory_time_dependent(
    path: List[Tuple[int, int]],
    lats: np.ndarray,
    lons: np.ndarray,
    departure_time: pd.Timestamp,
    time_values_ns: np.ndarray,
    hs_cube: np.ndarray,
    tp_cube: np.ndarray,
    vmdr_sin_cube: np.ndarray,
    vmdr_cos_cube: np.ndarray,
    wave_limit: float,
    safety_wave_weight: float,
    short_wave_penalty_weight: float,
    above_hs_limit_penalty_weight: float,
    fuel_wave_weight: float,
    fuel_steepness_weight: float,
    ship_speed_knots: float,
    min_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
) -> pd.DataFrame:
    """Build a cumulative time trajectory using evolving wave conditions."""
    if not path:
        return pd.DataFrame()

    rows = []
    total_time = 0.0
    i0, j0 = path[0]
    rows.append(
        {
            "cum_time_h": 0.0,
            "lat": float(lats[i0]),
            "lon": float(lons[j0]),
            "segment_index": 0,
            "hs_m": np.nan,
            "tp_s": np.nan,
            "wave_from_deg": np.nan,
            "heading_deg": np.nan,
            "wave_relative_angle_deg": np.nan,
            "effective_speed_knots": np.nan,
            "fuel_t": 0.0,
            "co2_t": 0.0,
            "safety": 0.0,
        }
    )

    for seg_idx, ((i1, j1), (i2, j2)) in enumerate(zip(path[:-1], path[1:]), start=1):
        sampled = sample_edge_wave_at_time(
            i=i1,
            j=j1,
            ni=i2,
            nj=j2,
            departure_time=departure_time,
            hours_from_departure=total_time,
            time_values_ns=time_values_ns,
            hs_cube=hs_cube,
            tp_cube=tp_cube,
            vmdr_sin_cube=vmdr_sin_cube,
            vmdr_cos_cube=vmdr_cos_cube,
        )
        if sampled is None:
            raise RuntimeError("Route trajectory exceeded available wave data time range.")

        hs_local, tp_local, wave_from_local = sampled
        comp = local_segment_components(
            i=i1,
            j=j1,
            ni=i2,
            nj=j2,
            lats=lats,
            lons=lons,
            hs_local=hs_local,
            tp_local=tp_local,
            wave_from_local=wave_from_local,
            wave_limit=wave_limit,
            safety_wave_weight=safety_wave_weight,
            short_wave_penalty_weight=short_wave_penalty_weight,
            above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
            fuel_wave_weight=fuel_wave_weight,
            fuel_steepness_weight=fuel_steepness_weight,
            ship_speed_knots=ship_speed_knots,
            min_speed_knots=min_speed_knots,
            reference_speed_knots=reference_speed_knots,
            reference_power_kw=reference_power_kw,
            sfoc_g_per_kwh=sfoc_g_per_kwh,
            co2_emission_factor=co2_emission_factor,
            speed_loss_wave_weight=speed_loss_wave_weight,
            speed_loss_direction_weight=speed_loss_direction_weight,
            fuel_direction_weight=fuel_direction_weight,
            safety_direction_weight=safety_direction_weight,
        )
        total_time += comp["time"]
        rows.append(
            {
                "cum_time_h": float(total_time),
                "lat": float(lats[i2]),
                "lon": float(lons[j2]),
                "segment_index": seg_idx,
                "hs_m": float(comp["hs"]),
                "tp_s": float(comp["tp"]),
                "wave_from_deg": float(comp.get("wave_from_deg", np.nan)),
                "heading_deg": float(comp.get("heading_deg", np.nan)),
                "wave_relative_angle_deg": float(comp.get("wave_relative_angle_deg", np.nan)),
                "effective_speed_knots": float(comp["effective_speed"]),
                "fuel_t": float(comp["fuel"]),
                "co2_t": float(comp["co2"]),
                "safety": float(comp["safety"]),
            }
        )

    return pd.DataFrame(rows)


def interpolate_trajectory_position(
    trajectory_records: List[Dict[str, float]],
    elapsed_h: float,
) -> Dict[str, float | str]:
    """Interpolate vessel position along a stored route trajectory."""
    if not trajectory_records:
        return {"status": "no trajectory", "lat": np.nan, "lon": np.nan, "progress_pct": np.nan}

    df = pd.DataFrame(trajectory_records).sort_values("cum_time_h").reset_index(drop=True)
    times = df["cum_time_h"].to_numpy(dtype=float)
    eta_h = float(times[-1]) if len(times) else 0.0

    if elapsed_h <= 0 or len(df) == 1:
        row = df.iloc[0]
        return {
            "status": "departing",
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "progress_pct": 0.0,
            "eta_h": eta_h,
            "effective_speed_knots": np.nan,
            "hs_m": np.nan,
        }

    if elapsed_h >= eta_h:
        row = df.iloc[-1]
        return {
            "status": "arrived",
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "progress_pct": 100.0,
            "eta_h": eta_h,
            "effective_speed_knots": float(row.get("effective_speed_knots", np.nan)),
            "hs_m": float(row.get("hs_m", np.nan)),
        }

    next_idx = int(np.searchsorted(times, elapsed_h, side="right"))
    next_idx = min(max(next_idx, 1), len(df) - 1)
    prev_idx = next_idx - 1

    t0 = float(times[prev_idx])
    t1 = float(times[next_idx])
    frac = 0.0 if t1 <= t0 else (float(elapsed_h) - t0) / (t1 - t0)

    prev_row = df.iloc[prev_idx]
    next_row = df.iloc[next_idx]
    lat = float(prev_row["lat"] + frac * (next_row["lat"] - prev_row["lat"]))
    lon = float(prev_row["lon"] + frac * (next_row["lon"] - prev_row["lon"]))

    return {
        "status": "en route",
        "lat": lat,
        "lon": lon,
        "progress_pct": 100.0 * float(elapsed_h) / max(eta_h, 1e-9),
        "eta_h": eta_h,
        "effective_speed_knots": float(next_row.get("effective_speed_knots", np.nan)),
        "hs_m": float(next_row.get("hs_m", np.nan)),
    }


def partial_trajectory_points(
    trajectory_records: List[Dict[str, float]],
    elapsed_h: float,
) -> List[Tuple[float, float]]:
    """Return travelled route polyline points up to the selected elapsed time."""
    if not trajectory_records:
        return []

    df = pd.DataFrame(trajectory_records).sort_values("cum_time_h").reset_index(drop=True)
    points = [
        (float(row["lat"]), float(row["lon"]))
        for _, row in df[df["cum_time_h"] <= elapsed_h].iterrows()
    ]

    pos = interpolate_trajectory_position(trajectory_records, elapsed_h)
    if np.isfinite(pos.get("lat", np.nan)) and np.isfinite(pos.get("lon", np.nan)):
        current_point = (float(pos["lat"]), float(pos["lon"]))
        if not points or points[-1] != current_point:
            points.append(current_point)

    return points


def _json_safe(value):
    """Convert numpy/pandas values to JSON-safe Python objects."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def rgba_image_to_data_url(image: np.ndarray) -> str:
    """Convert an RGBA numpy image to a browser-ready PNG data URL."""
    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGBA").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def finite_percentile_range(fields: List[np.ndarray], lower: float = 2, upper: float = 98) -> Tuple[float, float]:
    """Return a stable colour range across all timeline snapshots."""
    values = []
    for field in fields:
        arr = np.asarray(field, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite.ravel())

    if not values:
        return np.nan, np.nan

    joined = np.concatenate(values)
    vmin = float(np.nanpercentile(joined, lower))
    vmax = float(np.nanpercentile(joined, upper))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(joined))
        vmax = float(np.nanmax(joined))

    if vmin == vmax:
        vmax = vmin + 1e-6

    return vmin, vmax


def build_live_timeline_payload(
    result: Dict,
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    slp_ds: Optional[xr.Dataset] = None,
) -> Dict:
    """Precompute map frames and route time series for a browser-side timeline."""
    trajectories = result.get("route_trajectories", {})
    metrics_df = result.get("metrics_df", pd.DataFrame())

    departure_ts = pd.Timestamp(result.get("departure_time", result.get("actual_time")))
    max_eta_h = float(metrics_df["ETA / time proxy (h)"].max()) if not metrics_df.empty else 0.0
    end_ts = departure_ts + pd.Timedelta(hours=max_eta_h)

    map_bounds = {
        "min_lat": DATA_BOUNDS["min_lat"],
        "max_lat": DATA_BOUNDS["max_lat"],
        "min_lon": DATA_BOUNDS["min_lon"],
        "max_lon": DATA_BOUNDS["max_lon"],
    }

    raw_times = pd.to_datetime(ds["time"].values)
    timeline_times = [pd.Timestamp(t) for t in raw_times if departure_ts <= pd.Timestamp(t) <= end_ts]
    if not timeline_times:
        timeline_times = [departure_ts]

    timeline_fields = []
    for ts in timeline_times:
        snap = ds.sel(time=np.datetime64(ts), method="nearest")
        wave_time = pd.Timestamp(snap["time"].values)
        hs = np.asarray(snap["VHM0"].values, dtype=float)
        tp = np.asarray(snap["VTPK"].values, dtype=float)
        short_wave = hs / np.maximum(tp, 0.1)
        slp_contours = slp_contours_for_time(
            slp_ds=slp_ds,
            target_time=wave_time,
            map_bounds=map_bounds,
        )
        timeline_fields.append(
            {
                "label": wave_time.strftime("%Y-%m-%d %H:%M"),
                "elapsed_h": max(0.0, (wave_time - departure_ts).total_seconds() / 3600.0),
                "hs": hs,
                "tp": tp,
                "short_wave": short_wave,
                "slp_contours": slp_contours,
            }
        )

    layer_specs = {
        "hs": {
            "label": "Hs, significant wave height (m)",
            "cmap": "turbo",
            "fields": [f["hs"] for f in timeline_fields],
        },
        "tp": {
            "label": "Tp, peak wave period (s)",
            "cmap": "viridis",
            "fields": [f["tp"] for f in timeline_fields],
        },
        "short_wave": {
            "label": "Short-wave proxy, Hs/Tp",
            "cmap": "plasma",
            "fields": [f["short_wave"] for f in timeline_fields],
        },
    }

    colour_ranges = {
        key: finite_percentile_range(spec["fields"])
        for key, spec in layer_specs.items()
    }

    wave_snapshots = []
    for frame in timeline_fields:
        layers = {}
        for key, spec in layer_specs.items():
            vmin, vmax = colour_ranges[key]
            leaflet_field, bounds = resample_field_for_leaflet(
                field=frame[key],
                lats=lats,
                lons=lons,
            )
            image, used_vmin, used_vmax = field_to_rgba_image(
                field=leaflet_field,
                cmap_name=spec["cmap"],
                alpha=0.55,
                vmin=vmin,
                vmax=vmax,
            )
            layers[key] = {
                "label": spec["label"],
                "url": rgba_image_to_data_url(image),
                "bounds": bounds,
                "vmin": used_vmin,
                "vmax": used_vmax,
            }

        wave_snapshots.append(
            {
                "label": frame["label"],
                "elapsed_h": frame["elapsed_h"],
                "layers": layers,
                "slpContours": frame.get("slp_contours", {"available": False, "label": None, "lines": []}),
            }
        )

    route_payload = []
    all_points = [result["start"], result["end"]]

    for objective, latlon_path in result.get("routes", {}).items():
        trajectory_records = trajectories.get(objective, [])
        trajectory = []
        for row in trajectory_records:
            t = float(row.get("cum_time_h", np.nan))
            hs = row.get("hs_m", np.nan)
            tp = row.get("tp_s", np.nan)
            speed = row.get("effective_speed_knots", np.nan)
            wave_from = row.get("wave_from_deg", np.nan)
            heading = row.get("heading_deg", np.nan)
            wave_rel = row.get("wave_relative_angle_deg", np.nan)
            trajectory.append(
                {
                    "t": t,
                    "lat": float(row.get("lat", np.nan)),
                    "lon": float(row.get("lon", np.nan)),
                    "hs": float(hs) if hs is not None and np.isfinite(float(hs)) else np.nan,
                    "tp": float(tp) if tp is not None and np.isfinite(float(tp)) else np.nan,
                    "speed": float(speed) if speed is not None and np.isfinite(float(speed)) else np.nan,
                    "waveFrom": float(wave_from) if wave_from is not None and np.isfinite(float(wave_from)) else np.nan,
                    "heading": float(heading) if heading is not None and np.isfinite(float(heading)) else np.nan,
                    "waveRel": float(wave_rel) if wave_rel is not None and np.isfinite(float(wave_rel)) else np.nan,
                }
            )

        path = [[float(lat), float(lon)] for lat, lon in latlon_path]
        all_points.extend([(float(lat), float(lon)) for lat, lon in latlon_path])
        route_payload.append(
            {
                "name": objective,
                "color": ROUTE_COLORS.get(objective, "black"),
                "path": path,
                "trajectory": trajectory,
            }
        )

    min_lat = min(p[0] for p in all_points)
    max_lat = max(p[0] for p in all_points)
    min_lon = min(p[1] for p in all_points)
    max_lon = max(p[1] for p in all_points)

    return _json_safe(
        {
            "departureLabel": departure_ts.strftime("%Y-%m-%d %H:%M"),
            "waveSnapshots": wave_snapshots,
            "routes": route_payload,
            "center": [float((result["start"][0] + result["end"][0]) / 2), float((result["start"][1] + result["end"][1]) / 2)],
            "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
            "start": {"label": result["start_label"], "latlon": [float(result["start"][0]), float(result["start"][1])]},
            "end": {"label": result["end_label"], "latlon": [float(result["end"][0]), float(result["end"][1])]},
        }
    )


def live_timeline_html(payload: Dict) -> str:
    """Return a self-contained browser-side Leaflet plus Plotly timeline."""
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    template = '''
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; }
    .panel { border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; background: white; }
    .controls { display: grid; grid-template-columns: auto 1fr auto auto auto; gap: 10px; align-items: center; margin-bottom: 12px; }
    .controls button, .controls select { border: 1px solid #d1d5db; border-radius: 8px; background: #ffffff; padding: 7px 10px; cursor: pointer; }
    .controls input[type="range"] { width: 100%; cursor: ew-resize; }
    .timeLabel { font-weight: 650; min-width: 170px; }
    #liveMap { height: 540px; border-radius: 12px; border: 1px solid #e5e7eb; }
    .plotGrid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
    .plotBox { height: 310px; border: 1px solid #e5e7eb; border-radius: 12px; }
    .status { margin-top: 12px; overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: left; white-space: nowrap; }
    th { background: #f9fafb; }
    .hint { color: #6b7280; font-size: 13px; margin-bottom: 8px; }
    @media (max-width: 850px) {
      .controls { grid-template-columns: 1fr; }
      .plotGrid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="panel">
    <div class="hint">Drag the timeline. The map and the four time series update in the browser while the mouse is moving. Use the mouse wheel over the slider to step backward or forward.</div>
    <div class="controls">
      <button id="prevBtn">◀</button>
      <input id="timeRange" type="range" min="0" max="0" value="0" step="1" />
      <button id="nextBtn">▶</button>
      <select id="waveLayerSelect"></select>
      <label style="font-size:13px;display:flex;align-items:center;gap:5px;white-space:nowrap;"><input id="slpToggle" type="checkbox" checked /> SLP contours</label>
      <div class="timeLabel" id="timeLabel"></div>
    </div>
    <div id="liveMap"></div>
    <div class="plotGrid">
      <div id="hsPlot" class="plotBox"></div>
      <div id="tpPlot" class="plotBox"></div>
      <div id="speedPlot" class="plotBox"></div>
      <div id="waveDirPlot" class="plotBox"></div>
    </div>
    <div id="statusTable" class="status"></div>
  </div>

<script>
const payload = __TIMELINE_PAYLOAD__;
const range = document.getElementById('timeRange');
const timeLabel = document.getElementById('timeLabel');
const layerSelect = document.getElementById('waveLayerSelect');
const slpToggle = document.getElementById('slpToggle');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

range.max = Math.max(0, payload.waveSnapshots.length - 1);

const layerKeys = Object.keys(payload.waveSnapshots[0].layers);
for (const key of layerKeys) {
  const opt = document.createElement('option');
  opt.value = key;
  opt.textContent = payload.waveSnapshots[0].layers[key].label;
  layerSelect.appendChild(opt);
}
layerSelect.value = 'hs';

const map = L.map('liveMap', { preferCanvas: true }).setView(payload.center, 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

L.marker(payload.start.latlon, {title: payload.start.label}).addTo(map).bindPopup('Start: ' + payload.start.label);
L.marker(payload.end.latlon, {title: payload.end.label}).addTo(map).bindPopup('End: ' + payload.end.label);

let currentLayerKey = layerSelect.value;
let currentIndex = 0;
let waveOverlay = null;
let slpLayerGroup = L.layerGroup().addTo(map);
const fullRouteLayers = {};
const travelledLayers = {};
const vesselMarkers = {};

for (const route of payload.routes) {
  fullRouteLayers[route.name] = L.polyline(route.path, {
    color: route.color,
    weight: 3,
    opacity: 0.65
  }).addTo(map).bindTooltip(route.name);

  travelledLayers[route.name] = L.polyline([], {
    color: route.color,
    weight: 7,
    opacity: 0.90
  }).addTo(map).bindTooltip('Travelled: ' + route.name);

  vesselMarkers[route.name] = L.circleMarker(route.path[0], {
    radius: 8,
    color: route.color,
    fillColor: route.color,
    fillOpacity: 0.95,
    weight: 2
  }).addTo(map).bindTooltip(route.name);
}

map.fitBounds(payload.bounds, { padding: [18, 18] });
setTimeout(() => map.invalidateSize(), 120);

function asNumber(x) {
  return (x === null || x === undefined || Number.isNaN(Number(x))) ? null : Number(x);
}

function fmt(x, digits = 2) {
  const y = asNumber(x);
  return y === null ? 'n/a' : y.toFixed(digits);
}

function metricSeries(route, key) {
  return (route.trajectory || [])
    .filter(p => asNumber(p[key]) !== null && asNumber(p.t) !== null)
    .map(p => ({x: Number(p.t), y: Number(p[key])}));
}

function interpolateMetric(route, elapsed, key) {
  const s = metricSeries(route, key);
  if (s.length === 0) return null;
  if (elapsed <= s[0].x) return s[0].y;
  if (elapsed >= s[s.length - 1].x) return s[s.length - 1].y;
  let idx = 1;
  while (idx < s.length && s[idx].x < elapsed) idx += 1;
  const a = s[idx - 1];
  const b = s[idx];
  const frac = (b.x <= a.x) ? 0 : (elapsed - a.x) / (b.x - a.x);
  return a.y + frac * (b.y - a.y);
}

function interpolateTrajectory(route, elapsed) {
  const tr = route.trajectory || [];
  if (tr.length === 0) return null;
  if (elapsed <= tr[0].t || tr.length === 1) {
    const p = tr[0];
    return {lat: p.lat, lon: p.lon, hs: p.hs, tp: p.tp, speed: p.speed, waveFrom: p.waveFrom, heading: p.heading, waveRel: p.waveRel, progress: 0, status: 'departing'};
  }
  const last = tr[tr.length - 1];
  if (elapsed >= last.t) {
    return {lat: last.lat, lon: last.lon, hs: last.hs, tp: last.tp, speed: last.speed, waveFrom: last.waveFrom, heading: last.heading, waveRel: last.waveRel, progress: 100, status: 'arrived'};
  }
  let idx = 1;
  while (idx < tr.length && tr[idx].t < elapsed) idx += 1;
  const a = tr[idx - 1];
  const b = tr[idx];
  const frac = (b.t <= a.t) ? 0 : (elapsed - a.t) / (b.t - a.t);
  const lat = a.lat + frac * (b.lat - a.lat);
  const lon = a.lon + frac * (b.lon - a.lon);
  const hs = interpolateMetric(route, elapsed, 'hs');
  const tp = interpolateMetric(route, elapsed, 'tp');
  const speed = interpolateMetric(route, elapsed, 'speed');
  const waveFrom = interpolateMetric(route, elapsed, 'waveFrom');
  const heading = interpolateMetric(route, elapsed, 'heading');
  const waveRel = interpolateMetric(route, elapsed, 'waveRel');
  const progress = 100 * elapsed / Math.max(last.t, 1e-9);
  return {lat, lon, hs, tp, speed, waveFrom, heading, waveRel, progress, status: 'en route'};
}

function travelledPoints(route, elapsed) {
  const tr = route.trajectory || [];
  if (tr.length === 0) return [];
  const pts = [];
  for (const p of tr) {
    if (p.t <= elapsed) pts.push([p.lat, p.lon]);
  }
  const pos = interpolateTrajectory(route, elapsed);
  if (pos) pts.push([pos.lat, pos.lon]);
  return pts;
}

function plotMetric(divId, key, title, yLabel, elapsed, options = {}) {
  const traces = [];
  for (const route of payload.routes) {
    const s = metricSeries(route, key);
    traces.push({
      x: s.map(p => p.x),
      y: s.map(p => p.y),
      mode: 'lines',
      name: route.name,
      line: {color: route.color, width: 2},
      hovertemplate: route.name + '<br>Elapsed %{x:.1f} h<br>' + yLabel + ': %{y:.3f}<extra></extra>'
    });
    const currentY = interpolateMetric(route, elapsed, key);
    if (currentY !== null) {
      traces.push({
        x: [elapsed],
        y: [currentY],
        mode: 'markers',
        name: route.name + ' current',
        marker: {color: route.color, size: 11, line: {color: '#111827', width: 1}},
        showlegend: false,
        hovertemplate: route.name + ' current<br>Elapsed %{x:.1f} h<br>' + yLabel + ': %{y:.3f}<extra></extra>'
      });
    }
  }
  const maxEta = Math.max(...payload.routes.map(r => (r.trajectory && r.trajectory.length ? r.trajectory[r.trajectory.length - 1].t : 0)));
  const yaxis = {title: yLabel, rangemode: options.yRange ? undefined : 'tozero'};
  if (options.yRange) yaxis.range = options.yRange;
  if (options.tickVals) {
    yaxis.tickvals = options.tickVals;
    yaxis.ticktext = options.tickText;
  }
  const layout = {
    title: {text: title, font: {size: 15}},
    margin: {l: 55, r: 20, t: 45, b: 45},
    xaxis: {title: 'Elapsed voyage time (h)', range: [0, Math.max(maxEta, elapsed, 1)]},
    yaxis: yaxis,
    legend: {orientation: 'h', y: -0.25},
    shapes: [{type: 'line', x0: elapsed, x1: elapsed, y0: 0, y1: 1, yref: 'paper', line: {color: '#111827', width: 1, dash: 'dot'}}],
    hovermode: 'closest'
  };
  Plotly.react(divId, traces, layout, {displayModeBar: false, responsive: true});
}

function updateWaveOverlay(idx) {
  const frame = payload.waveSnapshots[idx];
  const layer = frame.layers[currentLayerKey];
  if (waveOverlay !== null) map.removeLayer(waveOverlay);
  waveOverlay = L.imageOverlay(layer.url, layer.bounds, {opacity: 0.65, interactive: false}).addTo(map);
  waveOverlay.bringToBack();
}

function updateSlpContours(idx) {
  slpLayerGroup.clearLayers();
  if (!slpToggle.checked) return;
  const frame = payload.waveSnapshots[idx];
  const contours = frame.slpContours || {available: false, lines: []};
  if (!contours.available || !contours.lines || contours.lines.length === 0) return;

  for (const line of contours.lines) {
    const poly = L.polyline(line.coords, {
      color: '#111827',
      weight: 1.2,
      opacity: 0.72,
      dashArray: '4 3',
      interactive: false
    }).addTo(slpLayerGroup);

    if (line.coords.length > 4) {
      const mid = line.coords[Math.floor(line.coords.length / 2)];
      L.marker(mid, {
        icon: L.divIcon({
          className: 'slp-label',
          html: '<span style="background:rgba(255,255,255,0.74);border:1px solid #374151;border-radius:3px;padding:1px 3px;font-size:10px;color:#111827;">' + Number(line.value).toFixed(0) + '</span>',
          iconSize: [34, 14]
        }),
        interactive: false
      }).addTo(slpLayerGroup);
    }
  }
}

function updateStatusTable(elapsed) {
  let html = '<table><thead><tr><th>Route</th><th>Status</th><th>Progress</th><th>Ship lat</th><th>Ship lon</th><th>Experienced Hs</th><th>Experienced Tp</th><th>Speed</th><th>Wave from</th><th>Relative wave angle</th></tr></thead><tbody>';
  for (const route of payload.routes) {
    const pos = interpolateTrajectory(route, elapsed);
    if (!pos) continue;
    html += `<tr><td style="font-weight:650;color:${route.color}">${route.name}</td><td>${pos.status}</td><td>${fmt(pos.progress, 1)}%</td><td>${fmt(pos.lat, 3)}</td><td>${fmt(pos.lon, 3)}</td><td>${fmt(pos.hs, 2)} m</td><td>${fmt(pos.tp, 3)} s</td><td>${fmt(pos.speed, 2)} kn</td><td>${fmt(pos.waveFrom, 0)}°</td><td>${fmt(pos.waveRel, 0)}°</td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('statusTable').innerHTML = html;
}

function updateAll() {
  currentIndex = Number(range.value);
  const frame = payload.waveSnapshots[currentIndex];
  const elapsed = Number(frame.elapsed_h);
  timeLabel.textContent = frame.label + '  |  +' + elapsed.toFixed(1) + ' h';
  updateWaveOverlay(currentIndex);
  updateSlpContours(currentIndex);

  for (const route of payload.routes) {
    const pos = interpolateTrajectory(route, elapsed);
    if (!pos) continue;
    vesselMarkers[route.name].setLatLng([pos.lat, pos.lon]);
    vesselMarkers[route.name].setTooltipContent(route.name + ' | ' + pos.status + ' | Hs ' + fmt(pos.hs, 2) + ' m | Tp ' + fmt(pos.tp, 2) + ' s | Speed ' + fmt(pos.speed, 1) + ' kn | rel wave ' + fmt(pos.waveRel, 0) + '°');
    travelledLayers[route.name].setLatLngs(travelledPoints(route, elapsed));
  }

  plotMetric('hsPlot', 'hs', 'Experienced significant wave height along route', 'Hs (m)', elapsed);
  plotMetric('tpPlot', 'tp', 'Experienced peak wave period along route', 'Tp (s)', elapsed);
  plotMetric('speedPlot', 'speed', 'Effective ship speed along route', 'Speed (kn)', elapsed);
  plotMetric(
    'waveDirPlot',
    'waveRel',
    'Wave direction relative to ship heading',
    'Relative angle (°)',
    elapsed,
    {yRange: [0, 180], tickVals: [0, 45, 90, 135, 180], tickText: ['0 head', '45', '90 beam', '135', '180 following']}
  );
  updateStatusTable(elapsed);
}

range.addEventListener('input', updateAll);
layerSelect.addEventListener('change', () => { currentLayerKey = layerSelect.value; updateAll(); });
slpToggle.addEventListener('change', updateAll);
prevBtn.addEventListener('click', () => { range.value = Math.max(0, Number(range.value) - 1); updateAll(); });
nextBtn.addEventListener('click', () => { range.value = Math.min(Number(range.max), Number(range.value) + 1); updateAll(); });
range.addEventListener('wheel', (event) => {
  event.preventDefault();
  const direction = event.deltaY > 0 ? 1 : -1;
  range.value = Math.min(Number(range.max), Math.max(0, Number(range.value) + direction));
  updateAll();
}, {passive: false});

updateAll();
</script>
</body>
</html>
'''
    return template.replace("__TIMELINE_PAYLOAD__", data_json)


def add_timeline_result_view(result: Dict, ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray, slp_ds: Optional[xr.Dataset] = None):
    """Render an interactive voyage timeline that updates inside the browser."""
    trajectories = result.get("route_trajectories", {})
    metrics_df = result.get("metrics_df", pd.DataFrame())

    if not trajectories or metrics_df.empty or "ETA / time proxy (h)" not in metrics_df.columns:
        return

    st.subheader("Voyage timeline")
    st.caption(
        "This timeline is rendered inside the browser, so the vessel markers, wave layer, "
        "and four plots update while you drag the slider. Wave maps use the raw NetCDF time steps; "
        "vessel positions are interpolated along each computed route. If `data/slp_sample.nc` is available, "
        "mean sea-level pressure contours are overlaid on the wave map."
    )

    try:
        payload = build_live_timeline_payload(result=result, ds=ds, lats=lats, lons=lons, slp_ds=slp_ds)
        components.html(live_timeline_html(payload), height=1380, scrolling=True)
    except Exception as exc:
        st.warning(
            "The live browser timeline could not be rendered. "
            "The metrics table above is still available. "
            f"Details: {exc}"
        )


def build_graph(
    lats: np.ndarray,
    lons: np.ndarray,
    hs: np.ndarray,
    tp: np.ndarray,
    vmdr: np.ndarray,
    fuel_factor: np.ndarray,
    safety_risk: np.ndarray,
    valid_mask: np.ndarray,
    no_data_penalty: np.ndarray,
    objective_weights: Dict[str, float],
    ship_speed_knots: float,
    min_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
) -> nx.Graph:
    """
    Build an 8-neighbour grid graph.

    Edge objective:
        weight =
            fuel_weight * segment_fuel_tonnes
          + time_weight * segment_time_h
          + safety_weight * segment_safety_index
          + distance_weight * segment_distance_nm

    The weights are demo business weights, not calibrated costs.
    """
    G = nx.Graph()
    nlat, nlon = hs.shape

    neighbours = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    fuel_w = objective_weights.get("fuel", 0.0)
    time_w = objective_weights.get("time", 0.0)
    safety_w = objective_weights.get("safety", 0.0)
    distance_w = objective_weights.get("distance", 0.0)

    for i in range(nlat):
        for j in range(nlon):
            if not valid_mask[i, j]:
                continue

            G.add_node((i, j))

            for di, dj in neighbours:
                ni, nj = i + di, j + dj

                if ni < 0 or ni >= nlat or nj < 0 or nj >= nlon:
                    continue

                if not valid_mask[ni, nj]:
                    continue

                if not edge_stays_in_valid_water(i, j, ni, nj, valid_mask):
                    continue

                comp = segment_components(
                    i=i,
                    j=j,
                    ni=ni,
                    nj=nj,
                    lats=lats,
                    lons=lons,
                    hs=hs,
                    tp=tp,
                    vmdr=vmdr,
                    fuel_factor=fuel_factor,
                    safety_risk=safety_risk,
                    ship_speed_knots=ship_speed_knots,
                    min_speed_knots=min_speed_knots,
                    reference_speed_knots=reference_speed_knots,
                    reference_power_kw=reference_power_kw,
                    sfoc_g_per_kwh=sfoc_g_per_kwh,
                    co2_emission_factor=co2_emission_factor,
                    speed_loss_wave_weight=speed_loss_wave_weight,
                    speed_loss_direction_weight=speed_loss_direction_weight,
                    fuel_direction_weight=fuel_direction_weight,
                    safety_direction_weight=safety_direction_weight,
                )

                weight = (
                    fuel_w * comp["fuel"]
                    + time_w * comp["time"]
                    + safety_w * comp["safety"]
                    + distance_w * comp["distance_nm"]
                )

                # Always discourage paths from hugging land/no-data boundaries.
                # This is separate from the selected objective because it is a
                # routing validity guardrail, not a commercial preference.
                local_no_data_penalty = float(
                    0.5 * (no_data_penalty[i, j] + no_data_penalty[ni, nj])
                    * comp["distance_nm"]
                )
                weight += NO_DATA_PROXIMITY_PENALTY_PER_NM * local_no_data_penalty

                G.add_edge(
                    (i, j),
                    (ni, nj),
                    weight=float(weight),
                    distance_km=float(comp["distance_km"]),
                    distance_nm=float(comp["distance_nm"]),
                    fuel=float(comp["fuel"]),
                    co2=float(comp["co2"]),
                    time=float(comp["time"]),
                    safety=float(comp["safety"]),
                )

    return G



def route_metrics(
    path: List[Tuple[int, int]],
    lats: np.ndarray,
    lons: np.ndarray,
    hs: np.ndarray,
    tp: np.ndarray,
    vmdr: np.ndarray,
    fuel_factor: np.ndarray,
    safety_risk: np.ndarray,
    wave_limit: float,
    ship_speed_knots: float,
    min_speed_knots: float,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
) -> Dict[str, float]:
    hs_values = np.array([hs[i, j] for i, j in path], dtype=float)
    tp_values = np.array([tp[i, j] for i, j in path], dtype=float)
    fuel_factor_values = np.array([fuel_factor[i, j] for i, j in path], dtype=float)
    safety_risk_values = np.array([safety_risk[i, j] for i, j in path], dtype=float)

    total_distance_km = 0.0
    total_distance_nm = 0.0
    total_fuel_tonnes = 0.0
    total_co2_tonnes = 0.0
    total_time = 0.0
    total_safety = 0.0
    total_power_time = 0.0

    head_sea_segments = 0
    beam_sea_segments = 0
    following_sea_segments = 0

    effective_speeds = []

    for (i1, j1), (i2, j2) in zip(path[:-1], path[1:]):
        comp = segment_components(
            i=i1,
            j=j1,
            ni=i2,
            nj=j2,
            lats=lats,
            lons=lons,
            hs=hs,
            tp=tp,
            vmdr=vmdr,
            fuel_factor=fuel_factor,
            safety_risk=safety_risk,
            ship_speed_knots=ship_speed_knots,
            min_speed_knots=min_speed_knots,
            reference_speed_knots=reference_speed_knots,
            reference_power_kw=reference_power_kw,
            sfoc_g_per_kwh=sfoc_g_per_kwh,
            co2_emission_factor=co2_emission_factor,
            speed_loss_wave_weight=speed_loss_wave_weight,
            speed_loss_direction_weight=speed_loss_direction_weight,
            fuel_direction_weight=fuel_direction_weight,
            safety_direction_weight=safety_direction_weight,
        )

        total_distance_km += comp["distance_km"]
        total_distance_nm += comp["distance_nm"]
        total_fuel_tonnes += comp["fuel"]
        total_co2_tonnes += comp["co2"]
        total_time += comp["time"]
        total_safety += comp["safety"]
        total_power_time += comp["engine_power_kw"] * comp["time"]
        effective_speeds.append(comp["effective_speed"])

        heading = bearing_deg(lats[i1], lons[j1], lats[i2], lons[j2])
        wave_from = float(vmdr[i1, j1])
        category = classify_wave_relative_to_heading(wave_from, heading)

        if category == "head":
            head_sea_segments += 1
        elif category == "beam":
            beam_sea_segments += 1
        else:
            following_sea_segments += 1

    nseg = max(1, len(path) - 1)
    mean_engine_power_kw = total_power_time / total_time if total_time > 0 else np.nan

    return {
        "Distance (km)": total_distance_km,
        "Distance (nm)": total_distance_nm,
        "ETA / time proxy (h)": total_time,
        "Mean effective speed (knots)": float(np.nanmean(effective_speeds)) if effective_speeds else np.nan,
        "Estimated fuel consumption (t)": total_fuel_tonnes,
        "Estimated CO₂ emissions (t)": total_co2_tonnes,
        "Mean engine power (kW)": mean_engine_power_kw,
        "Safety risk index": total_safety,
        "Mean Hs (m)": float(np.nanmean(hs_values)),
        "P90 Hs (m)": float(np.nanpercentile(hs_values, 90)),
        "Max Hs (m)": float(np.nanmax(hs_values)),
        f"Route cells Hs > {wave_limit:.1f}m (%)": float(np.mean(hs_values > wave_limit) * 100),
        "Mean Tp (s)": float(np.nanmean(tp_values)),
        "Mean weather power factor": float(np.nanmean(fuel_factor_values)),
        "Mean safety risk": float(np.nanmean(safety_risk_values)),
        "Head-sea segments (%)": 100 * head_sea_segments / nseg,
        "Beam-sea segments (%)": 100 * beam_sea_segments / nseg,
        "Following-sea segments (%)": 100 * following_sea_segments / nseg,
    }



def add_route_to_map(
    m: folium.Map,
    latlon_path: List[Tuple[float, float]],
    label: str,
    color: str,
    weight: int = 5,
):
    folium.PolyLine(
        locations=latlon_path,
        tooltip=label,
        color=color,
        weight=weight,
        opacity=0.9,
    ).add_to(m)


def add_port_marker(
    m: folium.Map,
    point: Tuple[float, float],
    label: str,
    color: str,
):
    folium.Marker(
        location=point,
        tooltip=label,
        popup=label,
        icon=folium.Icon(color=color, icon="flag"),
    ).add_to(m)



def coord_outer_bounds(coords: np.ndarray) -> Tuple[float, float]:
    """Return approximate outer bounds for 1D coordinate centres."""
    vals = np.asarray(coords, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan, np.nan

    if vals.size == 1:
        return float(vals[0] - 0.5), float(vals[0] + 0.5)

    vals = np.sort(vals)
    first_spacing = vals[1] - vals[0]
    last_spacing = vals[-1] - vals[-2]
    return float(vals[0] - 0.5 * first_spacing), float(vals[-1] + 0.5 * last_spacing)


WEB_MERCATOR_MAX_LAT = 85.05112878


def lat_to_web_mercator_y(lat: np.ndarray | float) -> np.ndarray | float:
    """Latitude in degrees to Web Mercator y, without Earth-radius scaling."""
    lat_arr = np.clip(lat, -WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT)
    rad = np.radians(lat_arr)
    return np.log(np.tan(np.pi / 4.0 + rad / 2.0))


def web_mercator_y_to_lat(y: np.ndarray | float) -> np.ndarray | float:
    """Web Mercator y, without Earth-radius scaling, back to latitude degrees."""
    return np.degrees(2.0 * np.arctan(np.exp(y)) - np.pi / 2.0)


def resample_field_for_leaflet(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Resample a lat/lon field onto a Web-Mercator-aligned image grid.

    Folium/Leaflet places ImageOverlay in Web Mercator. If we pass a raw
    latitude-linear image over a large north-south domain, the raster and the
    basemap can look shifted because Web Mercator is non-linear in latitude.
    This function samples the field at row centres that are evenly spaced in
    Web Mercator y, so the overlay aligns with the Leaflet basemap.
    """
    arr = np.asarray(field, dtype=float)

    lat_min_edge, lat_max_edge = coord_outer_bounds(lats)
    lon_min_edge, lon_max_edge = coord_outer_bounds(lons)

    lat_min_edge = max(lat_min_edge, -WEB_MERCATOR_MAX_LAT)
    lat_max_edge = min(lat_max_edge, WEB_MERCATOR_MAX_LAT)

    nlat, nlon = arr.shape

    y_north = lat_to_web_mercator_y(lat_max_edge)
    y_south = lat_to_web_mercator_y(lat_min_edge)

    y_edges = np.linspace(y_north, y_south, nlat + 1)
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    target_lats = web_mercator_y_to_lat(y_centres)

    lon_edges = np.linspace(lon_min_edge, lon_max_edge, nlon + 1)
    target_lons = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Nearest-neighbour resampling preserves the land/no-data mask instead of
    # smearing valid ocean values across coastlines.
    da = xr.DataArray(
        arr,
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
    )
    sampled = da.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="nearest",
    ).values

    bounds = [
        [float(lat_min_edge), float(lon_min_edge)],
        [float(lat_max_edge), float(lon_max_edge)],
    ]

    return sampled, bounds


def field_to_rgba_image(
    field: np.ndarray,
    cmap_name: str,
    alpha: float = 0.55,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Tuple[np.ndarray, float, float]:
    """Convert a north-to-south 2D field into an RGBA image for Folium."""
    arr = np.asarray(field, dtype=float)
    valid = np.isfinite(arr)

    if not valid.any():
        empty = np.zeros((*arr.shape, 4), dtype=np.uint8)
        return empty, np.nan, np.nan

    if vmin is None:
        vmin = float(np.nanpercentile(arr, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 98))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(arr[valid]))
        vmax = float(np.nanmax(arr[valid]))

    if vmin == vmax:
        vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps[cmap_name]

    safe_arr = np.where(valid, arr, vmin)
    rgba = cmap(norm(safe_arr))
    rgba[..., 3] = np.where(valid, alpha, 0.0)

    rgba_uint8 = (rgba * 255).astype(np.uint8)
    return rgba_uint8, vmin, vmax


def add_field_overlay(
    m: folium.Map,
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    name: str,
    cmap_name: str,
    show: bool = False,
    alpha: float = 0.55,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Dict[str, float]:
    """Add a gridded metocean field as a toggleable Folium overlay."""
    leaflet_field, bounds = resample_field_for_leaflet(
        field=field,
        lats=lats,
        lons=lons,
    )

    image, used_vmin, used_vmax = field_to_rgba_image(
        field=leaflet_field,
        cmap_name=cmap_name,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )

    folium.raster_layers.ImageOverlay(
        image=image,
        bounds=bounds,
        name=name,
        opacity=1.0,
        interactive=False,
        cross_origin=False,
        zindex=1,
        show=show,
    ).add_to(m)

    finite_field = np.asarray(field, dtype=float)

    return {
        "layer": name,
        "min": float(np.nanmin(finite_field)),
        "p02": used_vmin,
        "p98": used_vmax,
        "max": float(np.nanmax(finite_field)),
    }


def add_wave_field_overlays(
    m: folium.Map,
    result: Dict,
    lats: np.ndarray,
    lons: np.ndarray,
) -> pd.DataFrame:
    """Add Hs, Tp, and short-wave proxy overlays to the route map."""
    wave_fields = result.get("wave_overlay_fields", {})
    stats = []

    if "hs" in wave_fields:
        stats.append(
            add_field_overlay(
                m=m,
                field=wave_fields["hs"],
                lats=lats,
                lons=lons,
                name="Significant wave height Hs (m)",
                cmap_name="turbo",
                show=True,
                alpha=0.50,
            )
        )

    if "tp" in wave_fields:
        stats.append(
            add_field_overlay(
                m=m,
                field=wave_fields["tp"],
                lats=lats,
                lons=lons,
                name="Peak wave period Tp (s)",
                cmap_name="viridis",
                show=False,
                alpha=0.50,
            )
        )

    if "short_wave_proxy" in wave_fields:
        stats.append(
            add_field_overlay(
                m=m,
                field=wave_fields["short_wave_proxy"],
                lats=lats,
                lons=lons,
                name="Short-wave proxy Hs/Tp",
                cmap_name="magma",
                show=False,
                alpha=0.50,
            )
        )

    if stats:
        folium.LayerControl(collapsed=False).add_to(m)
        return pd.DataFrame(stats).set_index("layer")

    return pd.DataFrame()


def format_time_options(times: np.ndarray) -> List[str]:
    return [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in times]


def get_objective_weights(selected_objective: str) -> Dict[str, float]:
    """
    Objective weights are intentionally simple and transparent.

    Fuel is now in tonnes, while time is in hours and safety remains an index.
    The weights below are demo weights to make route preferences visible; they
    are not calibrated charter-party, fuel-price, or risk-cost coefficients.
    """
    if selected_objective == "Distance-only":
        return {"distance": 1.0, "fuel": 0.0, "time": 0.0, "safety": 0.0}

    if selected_objective == "Fuel-oriented":
        return {"distance": 0.1, "fuel": 1000.0, "time": 5.0, "safety": 0.05}

    if selected_objective == "Safety-oriented":
        return {"distance": 0.1, "fuel": 100.0, "time": 2.0, "safety": 2.0}

    if selected_objective == "ETA-oriented":
        return {"distance": 0., "fuel": 0., "time": 1.0, "safety": 0.}
        #return {"distance": 0.2, "fuel": 100.0, "time": 200.0, "safety": 0.05}

    # Balanced
    return {"distance": 0.2, "fuel": 500.0, "time": 50.0, "safety": 0.8}



def calculate_routes(
    start: Tuple[float, float],
    end: Tuple[float, float],
    start_label: str,
    end_label: str,
    selected_time,
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    selected_objectives: List[str],
    ship_speed_knots: float,
    min_speed_knots: float,
    wave_limit: float,
    safety_wave_weight: float,
    short_wave_penalty_weight: float,
    above_hs_limit_penalty_weight: float,
    fuel_wave_weight: float,
    fuel_steepness_weight: float,
    speed_loss_wave_weight: float,
    speed_loss_direction_weight: float,
    fuel_direction_weight: float,
    safety_direction_weight: float,
    vessel_profile: str,
    fuel_type: str,
    reference_speed_knots: float,
    reference_power_kw: float,
    sfoc_g_per_kwh: float,
    co2_emission_factor: float,
    weather_sensitivity: float,
    routing_mode: str,
    time_bin_minutes: int,
    corridor_width_nm: float,
    forecast_horizon_h: float,
) -> Dict:
    departure_ts = pd.Timestamp(selected_time)
    snap = ds.sel(time=selected_time, method="nearest")
    actual_time = pd.Timestamp(snap["time"].values).strftime("%Y-%m-%d %H:%M")

    hs = snap["VHM0"].values
    tp = snap["VTPK"].values
    vmdr = snap["VMDR"].values

    fields = compute_cell_fields(
        hs=hs,
        tp=tp,
        wave_limit=wave_limit,
        safety_wave_weight=safety_wave_weight,
        short_wave_penalty_weight=short_wave_penalty_weight,
        above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
        fuel_wave_weight=fuel_wave_weight,
        fuel_steepness_weight=fuel_steepness_weight,
    )

    fuel_factor = fields["fuel_factor"]
    safety_risk = fields["safety_risk"]

    valid_mask = (
        np.isfinite(hs)
        & np.isfinite(tp)
        & np.isfinite(vmdr)
        & np.isfinite(fuel_factor)
        & np.isfinite(safety_risk)
    )

    no_data_penalty = no_data_proximity_penalty(valid_mask)

    start_node = nearest_ocean_cell(start[0], start[1], lats, lons, valid_mask)
    end_node = nearest_ocean_cell(end[0], end[1], lats, lons, valid_mask)

    routes = {}
    metrics = {}
    route_trajectories = {}
    routing_diagnostics = {}

    use_time_dependent = routing_mode.startswith("Time-dependent")

    # Static baseline route is used both by static routing and as the corridor
    # centreline for the time-dependent A* search.
    baseline_graph = build_graph(
        lats=lats,
        lons=lons,
        hs=hs,
        tp=tp,
        vmdr=vmdr,
        fuel_factor=fuel_factor,
        safety_risk=safety_risk,
        valid_mask=valid_mask,
        no_data_penalty=no_data_penalty,
        objective_weights=get_objective_weights("Distance-only"),
        ship_speed_knots=ship_speed_knots,
        min_speed_knots=min_speed_knots,
        reference_speed_knots=reference_speed_knots,
        reference_power_kw=reference_power_kw,
        sfoc_g_per_kwh=sfoc_g_per_kwh,
        co2_emission_factor=co2_emission_factor,
        speed_loss_wave_weight=speed_loss_wave_weight,
        speed_loss_direction_weight=speed_loss_direction_weight,
        fuel_direction_weight=fuel_direction_weight,
        safety_direction_weight=safety_direction_weight,
    )
    baseline_path = nx.shortest_path(
        baseline_graph,
        source=start_node,
        target=end_node,
        weight="weight",
    )

    if use_time_dependent:
        corridor_mask = build_route_corridor_mask(
            baseline_path=baseline_path,
            valid_mask=valid_mask,
            lats=lats,
            lons=lons,
            corridor_width_nm=corridor_width_nm,
        )
        corridor_mask[start_node] = True
        corridor_mask[end_node] = True

        hs_cube = np.asarray(ds["VHM0"].values, dtype=float)
        tp_cube = np.asarray(ds["VTPK"].values, dtype=float)
        vmdr_cube = np.asarray(ds["VMDR"].values, dtype=float)
        vmdr_rad = np.deg2rad(vmdr_cube)
        vmdr_sin_cube = np.sin(vmdr_rad)
        vmdr_cos_cube = np.cos(vmdr_rad)
        time_values_ns = np.asarray(ds["time"].values, dtype="datetime64[ns]").astype("int64")

        for objective in selected_objectives:
            weights = get_objective_weights(objective)
            try:
                path, diag = time_dependent_a_star_route(
                    start_node=start_node,
                    end_node=end_node,
                    lats=lats,
                    lons=lons,
                    valid_mask=valid_mask,
                    search_mask=corridor_mask,
                    no_data_penalty=no_data_penalty,
                    objective_weights=weights,
                    departure_time=departure_ts,
                    time_values_ns=time_values_ns,
                    hs_cube=hs_cube,
                    tp_cube=tp_cube,
                    vmdr_sin_cube=vmdr_sin_cube,
                    vmdr_cos_cube=vmdr_cos_cube,
                    max_horizon_h=forecast_horizon_h,
                    time_bin_minutes=time_bin_minutes,
                    wave_limit=wave_limit,
                    safety_wave_weight=safety_wave_weight,
                    short_wave_penalty_weight=short_wave_penalty_weight,
                    above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
                    fuel_wave_weight=fuel_wave_weight,
                    fuel_steepness_weight=fuel_steepness_weight,
                    ship_speed_knots=ship_speed_knots,
                    min_speed_knots=min_speed_knots,
                    reference_speed_knots=reference_speed_knots,
                    reference_power_kw=reference_power_kw,
                    sfoc_g_per_kwh=sfoc_g_per_kwh,
                    co2_emission_factor=co2_emission_factor,
                    speed_loss_wave_weight=speed_loss_wave_weight,
                    speed_loss_direction_weight=speed_loss_direction_weight,
                    fuel_direction_weight=fuel_direction_weight,
                    safety_direction_weight=safety_direction_weight,
                )
                diag["used_corridor_fallback"] = 0.0
            except nx.NetworkXNoPath:
                # If the corridor is too narrow, retry once on the full valid mask.
                path, diag = time_dependent_a_star_route(
                    start_node=start_node,
                    end_node=end_node,
                    lats=lats,
                    lons=lons,
                    valid_mask=valid_mask,
                    search_mask=valid_mask,
                    no_data_penalty=no_data_penalty,
                    objective_weights=weights,
                    departure_time=departure_ts,
                    time_values_ns=time_values_ns,
                    hs_cube=hs_cube,
                    tp_cube=tp_cube,
                    vmdr_sin_cube=vmdr_sin_cube,
                    vmdr_cos_cube=vmdr_cos_cube,
                    max_horizon_h=forecast_horizon_h,
                    time_bin_minutes=time_bin_minutes,
                    wave_limit=wave_limit,
                    safety_wave_weight=safety_wave_weight,
                    short_wave_penalty_weight=short_wave_penalty_weight,
                    above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
                    fuel_wave_weight=fuel_wave_weight,
                    fuel_steepness_weight=fuel_steepness_weight,
                    ship_speed_knots=ship_speed_knots,
                    min_speed_knots=min_speed_knots,
                    reference_speed_knots=reference_speed_knots,
                    reference_power_kw=reference_power_kw,
                    sfoc_g_per_kwh=sfoc_g_per_kwh,
                    co2_emission_factor=co2_emission_factor,
                    speed_loss_wave_weight=speed_loss_wave_weight,
                    speed_loss_direction_weight=speed_loss_direction_weight,
                    fuel_direction_weight=fuel_direction_weight,
                    safety_direction_weight=safety_direction_weight,
                )
                diag["used_corridor_fallback"] = 1.0

            routes[objective] = path_to_latlon(path, lats, lons)
            metrics[objective] = route_metrics_time_dependent(
                path=path,
                lats=lats,
                lons=lons,
                departure_time=departure_ts,
                time_values_ns=time_values_ns,
                hs_cube=hs_cube,
                tp_cube=tp_cube,
                vmdr_sin_cube=vmdr_sin_cube,
                vmdr_cos_cube=vmdr_cos_cube,
                wave_limit=wave_limit,
                safety_wave_weight=safety_wave_weight,
                short_wave_penalty_weight=short_wave_penalty_weight,
                above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
                fuel_wave_weight=fuel_wave_weight,
                fuel_steepness_weight=fuel_steepness_weight,
                ship_speed_knots=ship_speed_knots,
                min_speed_knots=min_speed_knots,
                reference_speed_knots=reference_speed_knots,
                reference_power_kw=reference_power_kw,
                sfoc_g_per_kwh=sfoc_g_per_kwh,
                co2_emission_factor=co2_emission_factor,
                speed_loss_wave_weight=speed_loss_wave_weight,
                speed_loss_direction_weight=speed_loss_direction_weight,
                fuel_direction_weight=fuel_direction_weight,
                safety_direction_weight=safety_direction_weight,
            )
            route_trajectories[objective] = route_trajectory_time_dependent(
                path=path,
                lats=lats,
                lons=lons,
                departure_time=departure_ts,
                time_values_ns=time_values_ns,
                hs_cube=hs_cube,
                tp_cube=tp_cube,
                vmdr_sin_cube=vmdr_sin_cube,
                vmdr_cos_cube=vmdr_cos_cube,
                wave_limit=wave_limit,
                safety_wave_weight=safety_wave_weight,
                short_wave_penalty_weight=short_wave_penalty_weight,
                above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
                fuel_wave_weight=fuel_wave_weight,
                fuel_steepness_weight=fuel_steepness_weight,
                ship_speed_knots=ship_speed_knots,
                min_speed_knots=min_speed_knots,
                reference_speed_knots=reference_speed_knots,
                reference_power_kw=reference_power_kw,
                sfoc_g_per_kwh=sfoc_g_per_kwh,
                co2_emission_factor=co2_emission_factor,
                speed_loss_wave_weight=speed_loss_wave_weight,
                speed_loss_direction_weight=speed_loss_direction_weight,
                fuel_direction_weight=fuel_direction_weight,
                safety_direction_weight=safety_direction_weight,
            ).to_dict("records")
            routing_diagnostics[objective] = diag

    else:
        for objective in selected_objectives:
            weights = get_objective_weights(objective)

            # Re-use the baseline distance graph when possible.
            if objective == "Distance-only":
                graph = baseline_graph
            else:
                graph = build_graph(
                    lats=lats,
                    lons=lons,
                    hs=hs,
                    tp=tp,
                    vmdr=vmdr,
                    fuel_factor=fuel_factor,
                    safety_risk=safety_risk,
                    valid_mask=valid_mask,
                    no_data_penalty=no_data_penalty,
                    objective_weights=weights,
                    ship_speed_knots=ship_speed_knots,
                    min_speed_knots=min_speed_knots,
                    reference_speed_knots=reference_speed_knots,
                    reference_power_kw=reference_power_kw,
                    sfoc_g_per_kwh=sfoc_g_per_kwh,
                    co2_emission_factor=co2_emission_factor,
                    speed_loss_wave_weight=speed_loss_wave_weight,
                    speed_loss_direction_weight=speed_loss_direction_weight,
                    fuel_direction_weight=fuel_direction_weight,
                    safety_direction_weight=safety_direction_weight,
                )

            path = nx.shortest_path(
                graph,
                source=start_node,
                target=end_node,
                weight="weight",
            )

            routes[objective] = path_to_latlon(path, lats, lons)
            metrics[objective] = route_metrics(
                path=path,
                lats=lats,
                lons=lons,
                hs=hs,
                tp=tp,
                vmdr=vmdr,
                fuel_factor=fuel_factor,
                safety_risk=safety_risk,
                wave_limit=wave_limit,
                ship_speed_knots=ship_speed_knots,
                min_speed_knots=min_speed_knots,
                reference_speed_knots=reference_speed_knots,
                reference_power_kw=reference_power_kw,
                sfoc_g_per_kwh=sfoc_g_per_kwh,
                co2_emission_factor=co2_emission_factor,
                speed_loss_wave_weight=speed_loss_wave_weight,
                speed_loss_direction_weight=speed_loss_direction_weight,
                fuel_direction_weight=fuel_direction_weight,
                safety_direction_weight=safety_direction_weight,
            )
            route_trajectories[objective] = route_trajectory_static(
                path=path,
                lats=lats,
                lons=lons,
                hs=hs,
                tp=tp,
                vmdr=vmdr,
                fuel_factor=fuel_factor,
                safety_risk=safety_risk,
                wave_limit=wave_limit,
                ship_speed_knots=ship_speed_knots,
                min_speed_knots=min_speed_knots,
                reference_speed_knots=reference_speed_knots,
                reference_power_kw=reference_power_kw,
                sfoc_g_per_kwh=sfoc_g_per_kwh,
                co2_emission_factor=co2_emission_factor,
                speed_loss_wave_weight=speed_loss_wave_weight,
                speed_loss_direction_weight=speed_loss_direction_weight,
                fuel_direction_weight=fuel_direction_weight,
                safety_direction_weight=safety_direction_weight,
            ).to_dict("records")

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")

    snapped_start = (float(lats[start_node[0]]), float(lons[start_node[1]]))
    snapped_end = (float(lats[end_node[0]]), float(lons[end_node[1]]))

    return {
        "routes": routes,
        "metrics_df": metrics_df,
        "start": start,
        "end": end,
        "snapped_start": snapped_start,
        "snapped_end": snapped_end,
        "start_label": start_label,
        "end_label": end_label,
        "actual_time": actual_time,
        "departure_time": departure_ts.strftime("%Y-%m-%d %H:%M"),
        "wave_limit": wave_limit,
        "ship_speed_knots": ship_speed_knots,
        "selected_objectives": selected_objectives,
        "vessel_profile": vessel_profile,
        "fuel_type": fuel_type,
        "reference_speed_knots": reference_speed_knots,
        "reference_power_kw": reference_power_kw,
        "sfoc_g_per_kwh": sfoc_g_per_kwh,
        "co2_emission_factor": co2_emission_factor,
        "weather_sensitivity": weather_sensitivity,
        "routing_mode": routing_mode,
        "time_bin_minutes": time_bin_minutes,
        "corridor_width_nm": corridor_width_nm,
        "forecast_horizon_h": forecast_horizon_h,
        "routing_diagnostics": routing_diagnostics,
        "route_trajectories": route_trajectories,
        "wave_overlay_fields": {
            "hs": hs,
            "tp": tp,
            "short_wave_proxy": fields["steepness"],
        },
    }



def display_result(result: Dict, slp_ds: Optional[xr.Dataset] = None):
    st.subheader("Optimisation result")

    st.info(
        f"Start snapped to nearest valid ocean grid cell: {result['snapped_start']}. "
        f"End snapped to: {result['snapped_end']}."
    )

    st.caption(
        "Routing guardrail active: diagonal moves that cut across land/no-data "
        "corners are blocked, and cells next to no-data areas receive an internal "
        "penalty. This keeps routes visually inside the available wave field."
    )

    if result.get("routing_mode", "Static departure snapshot").startswith("Time-dependent"):
        st.info(
            "Time-dependent A* active: each searched segment samples Hs, Tp and wave direction "
            "at the vessel's estimated arrival time. The search is restricted to a corridor "
            "around the distance-only route and the final metrics are re-evaluated with "
            "continuous accumulated voyage time."
        )
        diagnostics = result.get("routing_diagnostics", {})
        if diagnostics:
            diag_df = pd.DataFrame.from_dict(diagnostics, orient="index")
            with st.expander("Time-dependent A* diagnostics", expanded=False):
                st.dataframe(diag_df.style.format("{:.2f}"), use_container_width=True)

    metrics_df = result["metrics_df"]

    st.dataframe(
        metrics_df.style.format("{:.2f}"),
        use_container_width=True,
    )

    if "Distance-only" in metrics_df.index:
        shortest_nm = metrics_df.loc["Distance-only", "Distance (nm)"]
        longest_nm = metrics_df["Distance (nm)"].max()
        if shortest_nm > 0 and longest_nm / shortest_nm > MAX_ROUTE_DETOUR_FACTOR_FOR_WARNING:
            st.warning(
                "One route is taking a very large detour. This can happen when "
                "Vessel weather sensitivity is set very high, because the demo "
                "strongly penalises wave exposure. The route is still valid in the "
                "graph, but this is a stress-test setting rather than a calibrated "
                "vessel model."
            )

    # The live voyage timeline is the main result map.
    # It combines route geometry, vessel positions, and raw wave snapshots in one view.


    add_timeline_result_view(result, ds=ds, lats=lats, lons=lons, slp_ds=slp_ds)

    st.subheader("Trade-off summary")

    if "Distance-only" in metrics_df.index:
        baseline = metrics_df.loc["Distance-only"]

        summary_rows = []

        for objective in metrics_df.index:
            if objective == "Distance-only":
                continue

            row = metrics_df.loc[objective]

            distance_change = (
                row["Distance (nm)"] / baseline["Distance (nm)"] - 1
            ) * 100

            fuel_change = (
                row["Estimated fuel consumption (t)"] / baseline["Estimated fuel consumption (t)"] - 1
            ) * 100

            safety_change = (
                row["Safety risk index"] / baseline["Safety risk index"] - 1
            ) * 100 if baseline["Safety risk index"] != 0 else np.nan

            max_hs_change = row["Max Hs (m)"] - baseline["Max Hs (m)"]

            summary_rows.append(
                {
                    "Route": objective,
                    "Distance change vs shortest (%)": distance_change,
                    "Fuel change vs shortest (%)": fuel_change,
                    "Safety risk change vs shortest (%)": safety_change,
                    "Max Hs change vs shortest (m)": max_hs_change,
                }
            )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).set_index("Route")
            st.dataframe(summary_df.style.format("{:.2f}"), use_container_width=True)

    st.markdown(
        """
        **Interpretation:**  
        A safety-oriented route may be longer and sometimes less fuel-efficient, but it should reduce wave exposure.  
        A fuel-oriented route may accept more wave exposure if avoiding it requires a large detour.  
        This is exactly why voyage optimisation is a trade-off, not a single universal “best route.”
        """
    )

    st.warning(
        "This is a simplified historical replay prototype. "
        "It should not be used for navigation or formal emissions reporting. "
        "Fuel and CO₂ are estimated from assumed vessel profiles, not verified "
        "noon reports or a vessel-specific speed-power curve. A production system "
        "would need time-dependent forecasts, wind, current, bathymetry, official "
        "navigational constraints, traffic separation schemes, vessel-specific "
        "performance curves, and expert review."
    )


# ============================================================
# Main app
# ============================================================

st.title("🌊 Wave-Aware Voyage Optimisation Demo")
st.caption(
    "Historical replay prototype using Copernicus Marine wave reanalysis. "
    "This app compares distance, fuel-oriented, safety-oriented, ETA-oriented, "
    "and balanced routing objectives."
)

try:
    ds = load_dataset(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Cannot find data file: `{DATA_PATH}`. "
        "Put your NetCDF file in the `data/` folder or change `DATA_PATH`."
    )
    st.stop()

required_vars = ["VHM0", "VTPK", "VMDR"]
missing_vars = [v for v in required_vars if v not in ds.data_vars]

if missing_vars:
    st.error(f"Missing variables in NetCDF: {missing_vars}")
    st.write("Available variables:", list(ds.data_vars))
    st.stop()

lats = ds["latitude"].values
lons = ds["longitude"].values
times = ds["time"].values

slp_ds = None
if os.path.exists(SLP_DATA_PATH):
    try:
        slp_ds = load_slp_dataset(SLP_DATA_PATH)
    except Exception as exc:
        st.sidebar.warning(f"Could not load SLP overlay file `{SLP_DATA_PATH}`: {exc}")
else:
    st.sidebar.caption(
        f"Optional SLP overlay: place ERA5 mean sea-level pressure at `{SLP_DATA_PATH}` "
        "to show pressure contours on the live wave map."
    )


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Voyage setup")

use_manual = st.sidebar.checkbox("Use manual coordinates", value=False)

if use_manual:
    start_lat = st.sidebar.number_input("Start latitude", value=51.95, format="%.4f")
    start_lon = st.sidebar.number_input("Start longitude", value=4.14, format="%.4f")
    end_lat = st.sidebar.number_input("End latitude", value=57.70, format="%.4f")
    end_lon = st.sidebar.number_input("End longitude", value=11.97, format="%.4f")

    start_label = "Manual start"
    end_label = "Manual end"
    start = (start_lat, start_lon)
    end = (end_lat, end_lon)
else:
    port_names = list(PORTS_IN_DOMAIN.keys())

    start_label = st.sidebar.selectbox(
        "Departure port",
        port_names,
        index=port_names.index("Rotterdam") if "Rotterdam" in port_names else 0,
    )

    default_end = "Gothenburg" if "Gothenburg" in port_names else port_names[min(1, len(port_names) - 1)]

    end_label = st.sidebar.selectbox(
        "Arrival port",
        port_names,
        index=port_names.index(default_end),
    )

    start = PORTS_IN_DOMAIN[start_label]
    end = PORTS_IN_DOMAIN[end_label]

if start_label == end_label:
    st.warning("Please choose different departure and arrival ports.")
    st.stop()

st.sidebar.header("Historical replay time")

time_labels = format_time_options(times)

time_label = st.sidebar.selectbox(
    "Departure datetime / wave snapshot",
    time_labels,
    index=0,
)

selected_time = pd.Timestamp(time_label).to_datetime64()

st.sidebar.header("Routing time treatment")

routing_mode = st.sidebar.selectbox(
    "Routing mode",
    ["Static departure snapshot", "Time-dependent A* (experimental)"],
    index=0,
)

selected_ts = pd.Timestamp(selected_time)
last_data_ts = pd.Timestamp(times[-1])
available_horizon_h = max(
    0.0,
    (last_data_ts - selected_ts).total_seconds() / 3600.0,
)

if routing_mode.startswith("Time-dependent"):
    time_bin_minutes = st.sidebar.selectbox(
        "A* arrival-time bin",
        [30, 60, 90, 180],
        index=1,
        help=(
            "Smaller bins are more time-accurate but slower. The final route is "
            "re-evaluated with continuous accumulated time."
        ),
    )

    corridor_width_nm = st.sidebar.slider(
        "Search corridor width (nm)",
        min_value=50,
        max_value=300,
        value=DEFAULT_CORRIDOR_WIDTH_NM,
        step=25,
        help=(
            "The time-dependent search is restricted to a corridor around the "
            "distance-only route so it can run on a laptop."
        ),
    )

    max_horizon_slider = int(max(6, min(120, math.floor(available_horizon_h / 3) * 3)))
    default_horizon = int(max(6, min(DEFAULT_FORECAST_HORIZON_H, max_horizon_slider)))

    forecast_horizon_h = st.sidebar.slider(
        "Max forecast horizon (h)",
        min_value=6,
        max_value=max_horizon_slider,
        value=default_horizon,
        step=3,
    )

    if available_horizon_h < 12:
        st.sidebar.warning(
            "Limited wave data after the selected departure time. Choose an earlier "
            "departure time or use static routing if A* cannot reach the destination."
        )
else:
    time_bin_minutes = DEFAULT_TIME_BIN_MINUTES
    corridor_width_nm = DEFAULT_CORRIDOR_WIDTH_NM
    forecast_horizon_h = min(DEFAULT_FORECAST_HORIZON_H, max(6.0, available_horizon_h))

st.sidebar.header("Objectives")

objective_options = [
    "Distance-only",
    "Fuel-oriented",
    "Safety-oriented",
    "Balanced",
    "ETA-oriented",
]

selected_objectives = st.sidebar.multiselect(
    "Routes to calculate",
    objective_options,
    default=["Distance-only", "Fuel-oriented", "Safety-oriented", "Balanced"],
)

if not selected_objectives:
    st.warning("Please select at least one routing objective.")
    st.stop()

st.sidebar.header("Ship settings")

# Vessel profile and fuel type are intentionally implicit in the UI.
# They still exist internally because the fuel and CO2 calculations need
# reference power, reference speed, SFOC, and a fuel-to-CO2 factor.
vessel_profile = DEFAULT_VESSEL_PROFILE
fuel_type = DEFAULT_FUEL_TYPE

profile = VESSEL_PROFILES[vessel_profile]
reference_speed_knots = profile["reference_speed_knots"]
reference_power_kw = profile["reference_power_kw"]
sfoc_g_per_kwh = profile["sfoc_g_per_kwh"]
co2_emission_factor = CO2_EMISSION_FACTORS[fuel_type]

ship_speed_knots = st.sidebar.slider(
    "Nominal ship speed (knots)",
    min_value=5.0,
    max_value=25.0,
    value=float(reference_speed_knots),
    step=0.5,
)

st.sidebar.caption(
    "Fuel and CO₂ are estimated with one fixed representative vessel/fuel "
    "assumption, so the demo stays focused on the route trade-off rather "
    "than vessel specification choices."
)

with st.sidebar.expander("Internal fuel / CO₂ assumptions", expanded=False):
    st.write(f"**Vessel preset:** {vessel_profile}")
    st.write(f"**Reference power:** {reference_power_kw:,.0f} kW")
    st.write(f"**Reference speed:** {reference_speed_knots:.1f} kn")
    st.write(f"**SFOC:** {sfoc_g_per_kwh:.0f} g/kWh")
    st.write(f"**Fuel type:** {fuel_type}")
    st.write(f"**CO₂ factor:** {co2_emission_factor:.3f} t CO₂/t fuel")

# Fixed model guardrail: prevents unrealistically low effective speeds in bad seas.
min_speed_knots = DEFAULT_MIN_EFFECTIVE_SPEED_KNOTS

st.sidebar.header("Wave / safety settings")

wave_limit = st.sidebar.slider(
    "Operational Hs limit (m)",
    min_value=1.0,
    max_value=8.0,
    value=3.0,
    step=0.1,
)

# Fixed internal safety coefficients. The sidebar keeps only the operational
# Hs limit visible, so the demo stays easier to interpret.
safety_wave_weight = SAFETY_WAVE_WEIGHT
short_wave_penalty_weight = SHORT_WAVE_PENALTY_WEIGHT
above_hs_limit_penalty_weight = ABOVE_HS_LIMIT_PENALTY_WEIGHT

st.sidebar.caption(
    "Safety uses fixed internal coefficients for Hs, short-period seas, "
    "above-limit penalty, and wave direction effect. Only the operational "
    "Hs limit is user-facing."
)

# Fixed internal safety direction coefficient.
# The actual head/beam/following classification is still calculated from VMDR
# relative to vessel heading; this coefficient only controls how strongly that
# classification affects the safety proxy.
safety_direction_weight = SAFETY_DIRECTION_WEIGHT

st.sidebar.header("Vessel weather response")

weather_sensitivity = st.sidebar.slider(
    "Vessel weather sensitivity",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help=(
        "Scales wave-added power and wave-related speed loss. "
        "It does not change the calm-water engine power assumption; "
        "it only controls how strongly bad weather affects this vessel."
    ),
)

# Fixed internal fuel and speed-loss coefficients, scaled by one simple
# user-facing vessel sensitivity control.
fuel_wave_weight = FUEL_WAVE_WEIGHT_BASE * weather_sensitivity
fuel_steepness_weight = FUEL_SHORT_WAVE_WEIGHT_BASE * weather_sensitivity
fuel_direction_weight = FUEL_DIRECTION_WEIGHT_BASE * weather_sensitivity
speed_loss_wave_weight = SPEED_LOSS_WAVE_WEIGHT_BASE * weather_sensitivity
speed_loss_direction_weight = SPEED_LOSS_DIRECTION_WEIGHT_BASE * weather_sensitivity

st.sidebar.caption(
    "Weather sensitivity scales two simplified weather response terms: "
    "wave-related speed loss and wave-added power. The detailed coefficients "
    "are fixed internally."
)

calculate_clicked = st.sidebar.button("Calculate routes", type="primary")

if st.sidebar.button("Clear result"):
    st.session_state.route_result = None
    st.rerun()


# ============================================================
# Overview panel
# ============================================================

left, right = st.columns([2, 1])

with right:
    st.subheader("Selected voyage")
    st.write(f"**From:** {start_label}")
    st.write(f"**To:** {end_label}")
    st.write(f"**Selected time:** {time_label}")
    st.write(f"**Routing mode:** {routing_mode}")
    if routing_mode.startswith("Time-dependent"):
        st.write(f"**A* time bin:** {time_bin_minutes} min")
        st.write(f"**Corridor width:** {corridor_width_nm:.0f} nm")
    st.write(f"**Nominal speed:** {ship_speed_knots:.1f} knots")
    st.write("**Vessel / fuel assumptions:** fixed internally")
    st.write(f"**Operational Hs limit:** {wave_limit:.1f} m")
    st.write(f"**Weather sensitivity:** {weather_sensitivity:.1f}×")

    st.subheader("Model structure")
    st.code(
        "Fuel estimate:\n"
        "  calm_power = ref_power * (speed / ref_speed)^3\n"
        "  weather_power = calm_power * weather_power_factor\n"
        "  fuel_t = weather_power * SFOC * time_h / 1e6\n\n"
        "CO2 estimate:\n"
        "  co2_t = fuel_t * fuel_emission_factor\n\n"
        "Time estimate:\n"
        "  distance_nm / effective_speed\n\n"
        "Safety proxy:\n"
        "  distance_nm * safety_risk"
    )

with left:
    st.subheader("Voyage overview")

    center = [
        float((start[0] + end[0]) / 2),
        float((start[1] + end[1]) / 2),
    ]

    overview_map = folium.Map(location=center, zoom_start=5)

    add_port_marker(overview_map, start, f"Start: {start_label}", "green")
    add_port_marker(overview_map, end, f"End: {end_label}", "red")

    folium.Rectangle(
        bounds=[
            [DATA_BOUNDS["min_lat"], DATA_BOUNDS["min_lon"]],
            [DATA_BOUNDS["max_lat"], DATA_BOUNDS["max_lon"]],
        ],
        color="blue",
        weight=2,
        fill=False,
        tooltip="Data domain",
    ).add_to(overview_map)

    st_folium(overview_map, height=450, width=900, key="overview_map")


# ============================================================
# Calculation
# ============================================================

if calculate_clicked:
    with st.spinner("Calculating routes... This may take a moment for a large grid."):
        try:
            result = calculate_routes(
                start=start,
                end=end,
                start_label=start_label,
                end_label=end_label,
                selected_time=selected_time,
                ds=ds,
                lats=lats,
                lons=lons,
                selected_objectives=selected_objectives,
                ship_speed_knots=ship_speed_knots,
                min_speed_knots=min_speed_knots,
                wave_limit=wave_limit,
                safety_wave_weight=safety_wave_weight,
                short_wave_penalty_weight=short_wave_penalty_weight,
                above_hs_limit_penalty_weight=above_hs_limit_penalty_weight,
                fuel_wave_weight=fuel_wave_weight,
                fuel_steepness_weight=fuel_steepness_weight,
                speed_loss_wave_weight=speed_loss_wave_weight,
                speed_loss_direction_weight=speed_loss_direction_weight,
                fuel_direction_weight=fuel_direction_weight,
                safety_direction_weight=safety_direction_weight,
                vessel_profile=vessel_profile,
                fuel_type=fuel_type,
                reference_speed_knots=reference_speed_knots,
                reference_power_kw=reference_power_kw,
                sfoc_g_per_kwh=sfoc_g_per_kwh,
                co2_emission_factor=co2_emission_factor,
                weather_sensitivity=weather_sensitivity,
                routing_mode=routing_mode,
                time_bin_minutes=int(time_bin_minutes),
                corridor_width_nm=float(corridor_width_nm),
                forecast_horizon_h=float(forecast_horizon_h),
            )

            st.session_state.route_result = result

        except nx.NetworkXNoPath:
            st.error(
                "No ocean route found between the selected points. "
                "Try different ports, manual coordinates farther from land, or a larger domain."
            )

        except Exception as exc:
            st.error(f"Calculation failed: {exc}")


# ============================================================
# Display stored result
# ============================================================

if st.session_state.route_result is not None:
    display_result(st.session_state.route_result, slp_ds=slp_ds)
else:
    st.info("Choose voyage/settings in the sidebar, then click **Calculate routes**.")