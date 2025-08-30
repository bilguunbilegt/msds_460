import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD, value

# --- Parameters ---
FLEET_SIZE = 3528                    # taxis available per hour
TRIPS_PER_TAXI_PER_HOUR = 2.0        # r; tune from your data (1.5–2.5 typical)

# Load demand (typical day)
demand_df = pd.read_csv("hourly_zone_demand.csv")
demand_df["pickup_community_area"] = demand_df["pickup_community_area"].astype(int)
demand_df["hour"] = demand_df["hour"].astype(int)
demand_df["demand"] = demand_df["demand"].astype(float)

areas = sorted(demand_df["pickup_community_area"].unique())
hours = sorted(demand_df["hour"].unique())

# Helper to get demand(a,h)
dem = {(int(r.pickup_community_area), int(r.hour)): float(r.demand)
       for r in demand_df.itertuples(index=False)}

# Model
m = LpProblem("Taxi_Fleet_Allocation_MinUnmet", LpMinimize)

# Decision vars
x = {(a,h): LpVariable(f"x_{a}_{h}", lowBound=0, cat="Integer") for a in areas for h in hours}  # taxis
u = {(a,h): LpVariable(f"u_{a}_{h}", lowBound=0, cat="Continuous") for a in areas for h in hours}  # unmet trips

# Objective: minimize total unmet demand
m += lpSum(u[a,h] for a in areas for h in hours)

# Demand satisfaction with slack: r*x + u >= demand
r = TRIPS_PER_TAXI_PER_HOUR
for a in areas:
    for h in hours:
        m += r * x[a,h] + u[a,h] >= dem.get((a,h), 0.0), f"demand_cover_{a}_{h}"

# Hourly fleet capacity
for h in hours:
    m += lpSum(x[a,h] for a in areas) <= FLEET_SIZE, f"fleet_cap_hour_{h}"

# Solve
m.solve(PULP_CBC_CMD(msg=False))

print("Status:", LpStatus[m.status])
total_unmet = value(m.objective)
print("Total unmet trips (typical day):", round(total_unmet, 2))

# Build solution table
rows = []
for a in areas:
    for h in hours:
        taxis = int(round(value(x[a,h]) or 0))
        unmet = max(0.0, value(u[a,h]) or 0.0)
        demand = dem.get((a,h), 0.0)
        served = min(demand, r * taxis)
        rows.append({
            "pickup_community_area": a,
            "hour": h,
            "demand": demand,
            "assigned_taxis": taxis,
            "served_trips": served,
            "unmet_trips": unmet
        })

sol = pd.DataFrame(rows)
sol.to_csv("taxi_allocation_solution.csv", index=False)
print("Saved: taxi_allocation_solution.csv")
