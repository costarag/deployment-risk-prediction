"""
Synthetic data generator for the deployment risk prediction PoC.

Generates three datasets:
  - match_events: high-traffic windows (tournaments, weekend matches)
  - deploy_events: deployment activity with realistic daily/hourly patterns
  - incidents: deploy mistakes and load incidents correlated with traffic

To use your own data, replace or wrap these functions.
Each must return a DataFrame with the expected columns:

  match_events  -> timestamp, tournament, traffic_multiplier
  deploy_events -> timestamp, is_critical_fix, deploy_size
  incidents     -> timestamp, incident_type, severity
"""

import numpy as np
import pandas as pd


def _hours_between(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date, freq="h")


def generate_match_events(start_date, end_date):
    """
    Returns DataFrame: timestamp, tournament, traffic_multiplier
    """
    hours = _hours_between(start_date, end_date)
    rows = []

    windows = {
        "Brasileirao": {
            "days": {2, 3},
            "hours": range(19, 22),
            "base": 1.7,
            "prob": 0.80,
        },
        "Champions League": {
            "days": {2},
            "hours": range(15, 22),
            "base": 2.1,
            "prob": 0.65,
        },
        "Europa League": {
            "days": {3},
            "hours": range(15, 22),
            "base": 1.8,
            "prob": 0.60,
        },
        "Libertadores": {
            "days": {2, 3},
            "hours": range(19, 22),
            "base": 1.9,
            "prob": 0.75,
        },
        "Weekend matches": {
            "days": {5, 6},
            "hours": range(16, 21),
            "base": 2.0,
            "prob": 0.85,
        },
    }

    for ts in hours:
        dow = ts.dayofweek
        hour = ts.hour
        for tournament, cfg in windows.items():
            if dow in cfg["days"] and hour in cfg["hours"]:
                if np.random.rand() < cfg["prob"]:
                    multiplier = np.clip(
                        np.random.normal(cfg["base"], 0.20), 1.0, 3.2
                    )
                    rows.append(
                        {
                            "timestamp": ts,
                            "tournament": tournament,
                            "traffic_multiplier": float(multiplier),
                        }
                    )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def generate_deploy_events(start_date, end_date):
    """
    Returns DataFrame: timestamp, is_critical_fix, deploy_size
    """
    hours = _hours_between(start_date, end_date)
    rows = []

    def base_rate(dow, hour):
        if dow <= 3:
            if 9 <= hour < 12:
                return 0.9
            if 12 <= hour < 16:
                return 1.1
            if 16 <= hour < 18:
                return 1.7
            if 18 <= hour < 22:
                return 0.7
            return 0.08
        if dow == 4:
            if 10 <= hour < 15:
                return 0.55
            if 15 <= hour < 17:
                return 1.0
            if 17 <= hour < 20:
                return 0.45
            return 0.06
        return 0.03

    for hour_ts in hours:
        dow = hour_ts.dayofweek
        hour = hour_ts.hour
        n_deploys = np.random.poisson(base_rate(dow, hour))

        for _ in range(n_deploys):
            minute = int(np.random.randint(0, 60))
            ts = hour_ts + pd.Timedelta(minutes=minute)

            base_crit = 0.07
            if dow >= 5:
                base_crit = 0.22
            if dow == 4 and 15 <= hour <= 18:
                base_crit += 0.10

            is_critical_fix = bool(np.random.rand() < base_crit)

            if is_critical_fix:
                deploy_size = np.random.choice(["small", "medium"], p=[0.70, 0.30])
            else:
                deploy_size = np.random.choice(
                    ["small", "medium", "large"], p=[0.45, 0.40, 0.15]
                )

            rows.append(
                {
                    "timestamp": ts,
                    "is_critical_fix": is_critical_fix,
                    "deploy_size": str(deploy_size),
                }
            )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def generate_incidents(deploys_df, matches_df):
    """
    Returns DataFrame: timestamp, incident_type, severity
    """
    rows = []

    if matches_df.empty:
        hourly_match = pd.Series(dtype=float)
    else:
        hourly_match = (
            matches_df.assign(hour=lambda d: d["timestamp"].dt.floor("h"))
            .groupby("hour")["traffic_multiplier"]
            .sum()
            .sort_index()
        )

    for _, dep in deploys_df.iterrows():
        ts = dep["timestamp"]
        hour_ts = ts.floor("h")
        dow = ts.dayofweek
        hour = ts.hour

        near_hours = pd.date_range(
            hour_ts - pd.Timedelta(hours=2),
            hour_ts + pd.Timedelta(hours=2),
            freq="h",
        )
        near_match_intensity = float(
            hourly_match.reindex(near_hours, fill_value=0.0).max()
        )

        p = 0.010
        if 16 <= hour <= 18:
            p += 0.040
        if dow in (2, 3):
            p += 0.010
        if dow == 4 and 15 <= hour <= 18:
            p += 0.025
        if dep["is_critical_fix"]:
            p += 0.020
        if dep["deploy_size"] == "large":
            p += 0.030
        elif dep["deploy_size"] == "medium":
            p += 0.015

        p += min(0.08, near_match_intensity * 0.02)
        p = float(np.clip(p, 0.0, 0.95))

        if np.random.rand() < p:
            if p >= 0.12:
                severity = "high"
            elif p >= 0.07:
                severity = "medium"
            else:
                severity = "low"

            rows.append(
                {
                    "timestamp": ts,
                    "incident_type": "deploy_incident",
                    "severity": severity,
                }
            )

    for hour_ts, intensity in hourly_match.items():
        lam = max(0.0, 0.03 * intensity)
        n_load = np.random.poisson(lam)

        for _ in range(n_load):
            minute = int(np.random.randint(0, 60))
            ts = hour_ts + pd.Timedelta(minutes=minute)

            if intensity >= 4.5:
                severity = "high"
            elif intensity >= 3.0:
                severity = "medium"
            else:
                severity = "low"

            rows.append(
                {
                    "timestamp": ts,
                    "incident_type": "load_incident",
                    "severity": severity,
                }
            )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
