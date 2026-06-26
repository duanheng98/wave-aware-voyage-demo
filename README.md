# Wave-Aware Voyage Optimisation Demo

A Streamlit demo showing how wave conditions can influence voyage route choice, estimated fuel use, CO₂ emissions, ETA, and safety exposure.

The app compares different routing objectives:

* Distance-only
* Fuel-oriented
* Safety-oriented
* ETA-oriented
* Balanced

It is designed as a technical prototype for learning and interview discussion, not for operational navigation.

---

## Demo idea

The ocean domain is represented as a grid graph. Each valid ocean grid cell is a node, and neighbouring cells are connected as possible route segments.

For each segment, the app estimates:

* Distance
* Travel time
* Effective vessel speed
* Fuel consumption
* CO₂ emissions
* Wave-related safety exposure

The selected route objective determines how these components are weighted.

---

## Data

The deployed version uses small pre-processed NetCDF sample files:

```text
data/wave_sample.nc
data/slp_sample.nc   # optional
```

Required wave variables:

```text
VHM0   significant wave height
VTPK   peak wave period
VMDR   mean wave direction
```

The optional SLP file is used to show mean sea-level pressure contours on the map.

The sample dataset is cut from a larger NetCDF file by selecting only a short time period, a limited spatial domain, and the required variables.

---

## Routing modes

The app supports two routing modes:

1. **Static departure snapshot**
   Uses wave conditions at the selected departure time.

2. **Time-dependent A* routing**
   Samples wave conditions based on the vessel’s estimated arrival time at each route segment.

---

## Model structure

Simplified fuel estimate:

```text
calm_power = reference_power * (speed / reference_speed)^3
weather_power = calm_power * weather_power_factor
fuel_t = weather_power * SFOC * time_h / 1e6
```

CO₂ estimate:

```text
co2_t = fuel_t * fuel_emission_factor
```

Safety proxy:

```text
safety_exposure = distance_nm * local_safety_risk
```

The safety proxy increases with higher wave height, short-period seas, waves above the operational Hs limit, and unfavourable wave direction relative to the vessel heading.

---

## Run locally

```bash
conda create -n route_demo python=3.11
conda activate route_demo
pip install -r requirements.txt
streamlit run app.py
```

---

## Deployment

For Streamlit Cloud, only commit the small sample data files.

Recommended `.gitignore`:

```gitignore
data/raw_wave.nc
data/raw_slp.nc
*.nc
!data/wave_sample.nc
!data/slp_sample.nc
__pycache__/
.env
.DS_Store
```

---

## Limitations

This is a simplified historical replay prototype.

It does not include full operational constraints such as wind, currents, bathymetry, traffic separation schemes, vessel-specific speed-power curves, or official navigational rules.

Fuel, CO₂, and safety values are simplified estimates and should not be used for navigation or formal reporting.

The purpose is to demonstrate the logic of weather-aware route optimisation and the trade-offs between distance, fuel, ETA, and safety.

