import numpy as np
import pandas as pd

# Parameters
num_nodes = 10000
field_size = 1000  # 1000m x 1000m

# Generate X, Y coordinates
x_positions = np.random.uniform(0, field_size, num_nodes)
y_positions = np.random.uniform(0, field_size, num_nodes)

# Assign Energy Levels
energy_levels = np.random.uniform(0.5, 2.0, num_nodes)

# Assign Roles (95% Normal, 5% CH)
roles = np.random.choice(['Normal', 'CH'], num_nodes, p=[0.95, 0.05])

# Assign Traffic Load (1-10 packets per round)
traffic_load = np.random.randint(1, 10, num_nodes)

# Create DataFrame
df = pd.DataFrame({
    'X': x_positions,
    'Y': y_positions,
    'Energy': energy_levels,
    'Role': roles,
    'TrafficLoad': traffic_load
})

# Save to CSV
df.to_csv("wsn_real_data.csv", index=False)
print("Dataset saved as 'wsn_real_data.csv'")