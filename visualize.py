import pandas as pd
import requests
import folium
from folium.plugins import TimestampedGeoJson

df = pd.read_csv("taxi_allocation_solution.csv")

# Fetch Chicago community areas GeoJSON
geo_url = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"
geojson = requests.get(geo_url).json()
area_features = {int(f["properties"]["area_numbe"]): f for f in geojson["features"]}

# Scale opacity by max assigned in the dataset (robust against different runs)
max_assigned = max(1.0, df["assigned_taxis"].max())

features = []
for hour in sorted(df["hour"].unique()):
    df_hour = df[df["hour"] == hour]
    for _, row in df_hour.iterrows():
        area = int(row["pickup_community_area"])
        feat = area_features.get(area)
        if not feat:  # skip areas missing from the map
            continue
        opacity = min(0.9, float(row["assigned_taxis"]) / max_assigned)
        features.append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": {
                "time": f"2024-01-01T{hour:02d}:00:00",
                "style": {
                    "color": "black",
                    "weight": 0.5,
                    "fillColor": "#ff0000",
                    "fillOpacity": opacity
                },
                "popup": (
                    f"Area: {area}"
                    f"<br>Hour: {hour}"
                    f"<br>Assigned Taxis: {int(row['assigned_taxis'])}"
                    f"<br>Demand: {row['demand']:.1f}"
                    f"<br>Served Trips: {row['served_trips']:.1f}"
                    f"<br>Unmet Trips: {row['unmet_trips']:.1f}"
                )
            }
        })

m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
TimestampedGeoJson(
    {"type": "FeatureCollection", "features": features},
    period="PT1H", add_last_point=False, auto_play=False, loop=False,
    max_speed=1, loop_button=True, date_options="HH:mm", time_slider_drag_update=True
).add_to(m)

m.save("taxi_allocation_timeslider.html")
print("Saved: taxi_allocation_timeslider.html")
