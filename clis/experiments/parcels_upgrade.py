import os
import numpy as np
import matplotlib.pyplot as plt
import folium
import plotly.express as px

# Paths
data_path = "data/orange_landvalue_per_acre.npy"
results_dir = "results/geo_visualizations"
os.makedirs(results_dir, exist_ok=True)

# Load data
data = np.load(data_path)
lon = data[:, 0]
lat = data[:, 1]
value = data[:, 2]

# Optional: cap extreme outliers for better visuals
upper_cap = np.percentile(value, 99)
value_vis = np.clip(value, None, upper_cap)

# =====================================================
# 1️⃣ 2D Matplotlib Scatter
# =====================================================

plt.figure(figsize=(10, 8))
plt.scatter(lon, lat, c=value_vis, cmap='viridis', s=2)
plt.colorbar(label="Land Value Per Acre")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Orange County Land Value Per Acre (2D)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "land_value_2d.png"), dpi=300)
plt.close()


# =====================================================
# 2️⃣ 3D Matplotlib Scatter
# =====================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(lon, lat, value_vis, c=value_vis, cmap='viridis', s=2)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Value Per Acre")
ax.set_title("Orange County Land Value Per Acre (3D)")
fig.colorbar(sc, shrink=0.5)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "land_value_3d.png"), dpi=300)
plt.close()


# =====================================================
# 3️⃣ Interactive Plotly 3D (Highly Recommended)
# =====================================================

fig_plotly = px.scatter_3d(
    x=lon,
    y=lat,
    z=value_vis,
    color=value_vis,
    color_continuous_scale="Viridis",
    opacity=0.7,
)

fig_plotly.update_layout(
    title="Interactive 3D Land Value Per Acre",
    scene=dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis_title="Value Per Acre"
    )
)

fig_plotly.write_html(os.path.join(results_dir, "interactive_3d_plotly.html"))


# =====================================================
# 4️⃣ Folium Map Overlay
# =====================================================

# Center map
center_lat = np.mean(lat)
center_lon = np.mean(lon)

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

# Normalize values for coloring
min_val = value_vis.min()
max_val = value_vis.max()

def normalize(val):
    return (val - min_val) / (max_val - min_val)

# Add points (use subset if dataset is huge)
sample_idx = np.random.choice(len(lat), size=min(10000, len(lat)), replace=False)

for i in sample_idx:
    norm_val = normalize(value_vis[i])
    color = plt.cm.viridis(norm_val)
    hex_color = '#%02x%02x%02x' % tuple(int(255*c) for c in color[:3])

    folium.CircleMarker(
        location=[lat[i], lon[i]],
        radius=2,
        color=hex_color,
        fill=True,
        fill_opacity=0.7,
        weight=0
    ).add_to(m)

m.save(os.path.join(results_dir, "land_value_folium_map.html"))

print("All visualizations saved to:", results_dir)
