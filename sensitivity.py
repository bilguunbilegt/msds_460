# sensitivity_analysis.py
# Requires: pandas, numpy, pulp  (CBC ships with PuLP; otherwise install coin-or-cbc)

import time
import numpy as np
import pandas as pd
import pulp as pl

# ---------- Data ----------
def load_demand(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    # normalize expected columns
    need = ["pickup_community_area", "hour", "demand"]
    for n in need:
        if n not in cols and n not in df.columns:
            raise ValueError(f"Missing column '{n}' in {path}. Found: {list(df.columns)}")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["pickup_community_area", "hour", "demand"]].copy()
    df["pickup_community_area"] = df["pickup_community_area"].astype(int)
    df["hour"] = df["hour"].astype(int)
    df["demand"] = df["demand"].astype(float)
    return df.sort_values(["hour", "pickup_community_area"]).reset_index(drop=True)

# ---------- MILP Solver ----------
def solve_allocation(demand_df: pd.DataFrame,
                     F: int,
                     r: float,
                     int_vars: bool = True,
                     hour_caps: dict | None = None,
                     time_limit: int | None = None,
                     msg: bool = False):
    """
    demand_df: columns [pickup_community_area, hour, demand]
    F: fleet size cap per hour (if hour_caps not provided)
    r: trips per taxi per hour
    hour_caps: optional dict {hour: cap} to override F for selected hours
    int_vars: if False, relax x to continuous (useful for speed or dual-like analysis)
    returns: dict with allocations df and summary metrics
    """
    A = sorted(demand_df["pickup_community_area"].unique().tolist())
    H = sorted(demand_df["hour"].unique().tolist())
    d = {(int(a), int(h)): float(val)
         for a, h, val in demand_df[["pickup_community_area", "hour", "demand"]].itertuples(index=False, name=None)}
    if hour_caps is None:
        hour_caps = {h: int(F) for h in H}
    else:
        # fill missing hours with F
        hour_caps = {int(h): int(hour_caps.get(h, F)) for h in H}

    # Build model
    prob = pl.LpProblem(name="taxi_allocation_min_unmet", sense=pl.LpMinimize)

    # Decision vars
    x_cat = pl.LpInteger if int_vars else pl.LpContinuous
    x = pl.LpVariable.dicts("x", ((a, h) for a in A for h in H), lowBound=0, cat=x_cat)
    u = pl.LpVariable.dicts("u", ((a, h) for a in A for h in H), lowBound=0, cat=pl.LpContinuous)

    # Objective: minimize total unmet trips
    prob += pl.lpSum(u[(a, h)] for a in A for h in H)

    # Demand coverage constraints
    for a in A:
        for h in H:
            prob += r * x[(a, h)] + u[(a, h)] >= d[(a, h)], f"cover_a{a}_h{h}"

    # Hourly fleet caps
    for h in H:
        prob += pl.lpSum(x[(a, h)] for a in A) <= hour_caps[h], f"fleetcap_h{h}"

    # Solve
    solver = pl.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    t0 = time.time()
    status = prob.solve(solver)
    runtime = time.time() - t0
    status_str = pl.LpStatus[status]

    # Extract solution
    sol = []
    for a in A:
        for h in H:
            xv = x[(a, h)].value() if x[(a, h)].value() is not None else 0.0
            uv = u[(a, h)].value() if u[(a, h)].value() is not None else 0.0
            dv = d[(a, h)]
            served = min(dv, r * xv)
            sol.append((a, h, dv, xv, served, uv))
    sol_df = pd.DataFrame(sol, columns=["pickup_community_area", "hour", "demand", "x_taxis", "served", "unmet"])

    # Metrics
    total_demand = float(sol_df["demand"].sum())
    total_unmet = float(sol_df["unmet"].sum())
    total_served = float(sol_df["served"].sum())
    served_pct = 0.0 if total_demand == 0 else total_served / total_demand

    summary = {
        "status": status_str,
        "runtime_sec": runtime,
        "objective_unmet": total_unmet,
        "demand_total": total_demand,
        "served_total": total_served,
        "served_pct": served_pct,
        "F": int(F),
        "r": float(r),
        "int_vars": bool(int_vars)
    }
    return {"allocations": sol_df, "summary": summary}

# ---------- Sensitivity grid ----------
def run_sensitivity(demand_df: pd.DataFrame,
                    F_grid: list[int],
                    r_grid: list[float],
                    int_vars: bool = True,
                    time_limit: int | None = None,
                    msg: bool = False) -> pd.DataFrame:
    rows = []
    for r in r_grid:
        for F in F_grid:
            res = solve_allocation(demand_df, F=F, r=r, int_vars=int_vars,
                                   time_limit=time_limit, msg=msg)
            rows.append(res["summary"])
    sens_df = pd.DataFrame(rows).sort_values(["r", "F"]).reset_index(drop=True)
    return sens_df

# ---------- Marginal benefit by hour (Δ unmet for +ΔF in one hour) ----------
def marginal_value_per_hour(demand_df: pd.DataFrame,
                            F: int,
                            r: float,
                            delta: int = 1,
                            int_vars: bool = True,
                            time_limit: int | None = None,
                            msg: bool = False) -> pd.DataFrame:
    base = solve_allocation(demand_df, F=F, r=r, int_vars=int_vars,
                            time_limit=time_limit, msg=msg)
    base_unmet = base["summary"]["objective_unmet"]
    hours = sorted(demand_df["hour"].unique().tolist())
    out = []
    for h in hours:
        caps = {hh: F for hh in hours}
        caps[h] = F + delta
        res_h = solve_allocation(demand_df, F=F, r=r, int_vars=int_vars,
                                 hour_caps=caps, time_limit=time_limit, msg=msg)
        du = base_unmet - res_h["summary"]["objective_unmet"]
        out.append({"hour": h, "deltaF_at_hour": delta, "unmet_reduction": du})
    return pd.DataFrame(out).sort_values("hour").reset_index(drop=True)

# ---------- Example usage ----------
if __name__ == "__main__":
    # 1) Load demand
    demand = load_demand("hourly_zone_demand.csv")

    # 2) Define grids
    F_base = 3528
    F_grid = [3000, 3300, 3528, 3800, 4200]       # tweak as needed
    r_grid = [1.5, 1.8, 2.0, 2.2, 2.5]            # tweak as needed

    # 3) Run sensitivity (integer x by default)
    sens = run_sensitivity(demand, F_grid=F_grid, r_grid=r_grid, int_vars=True, msg=False)
    sens["served_pct"] = (sens["served_pct"] * 100).round(2)
    print("\n=== Sensitivity Results (x integer) ===")
    print(sens[["r", "F", "status", "runtime_sec", "demand_total", "served_total", "objective_unmet", "served_pct"]])

    # Save table
    sens.to_csv("sensitivity_results.csv", index=False)
    print("\nSaved: sensitivity_results.csv")

    # 4) Marginal value per hour at the baseline (integer x)
    mv_hour = marginal_value_per_hour(demand, F=F_base, r=2.0, delta=1, int_vars=True, msg=False)
    print("\n=== Per-hour marginal benefit (Δ unmet for +1 taxi in that hour) ===")
    print(mv_hour)

    mv_hour.to_csv(f"marginal_value_per_hour_F{F_base}_r{2.0}.csv", index=False)
    print(f"\nSaved: marginal_value_per_hour_F{F_base}_r{2.0}.csv")
