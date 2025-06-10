import numpy as np
import pandas as pd
from config_file import *

df = pd.read_csv("datasets/MESOGEIA.csv")
df_weekly = pd.read_csv("datasets/weekly_alligned_scaled_profiles.csv")
print(df_weekly)

print(df.shape)
print(df.columns)
print(df[["TopNo", "Simulation"]])

# Allign simulations with indices
df["Simulation"] = df["Simulation"].apply(lambda x: x-2)
print(df[["TopNo", "Simulation"]])

# Indices
TEST_INDICES = [7307, 7500] # +7787

start_index = TEST_INDICES[0]

print(df.shape)

res = []
# 2 topologies
for topology_index in range(1,3):
    # 2 days of simulations
    if topology_index==1:
        a, b = 7307, 7307+96
    else:
        a, b = 7307+96, 7307+192
    for simulation in range(a, b): # 2 consecutive days
        tmp_df = df[(df["TopNo"]==topology_index) & (df["Simulation"]==simulation)]
        datetime_str = df_weekly.iloc[simulation]["Timestamp"]
        values_124 = tmp_df.iloc[124][["vm_pu", "va_degree", "Im_inj", "Ia_inj"]].values.tolist()
        res.append([datetime_str, 1, 1, values_124[0], topology_index])
        res.append([datetime_str, 1, 33, values_124[1], topology_index])
        res.append([datetime_str, 1, 2, values_124[2], topology_index])
        res.append([datetime_str, 1, 34, values_124[3], topology_index])

        values_127 = tmp_df.iloc[127][["vm_pu", "va_degree", "Im_inj", "Ia_inj"]].values.tolist()
        res.append([datetime_str, 2, 1, values_127[0], topology_index])
        res.append([datetime_str, 2, 33, values_127[1], topology_index])
        res.append([datetime_str, 2, 2, values_127[2], topology_index])
        res.append([datetime_str, 2, 34, values_127[3], topology_index])

        values_128 = tmp_df.iloc[128][["vm_pu", "va_degree", "Im_inj", "Ia_inj"]].values.tolist()
        res.append([datetime_str, 3, 1, values_128[0], topology_index])
        res.append([datetime_str, 3, 33, values_128[1], topology_index])
        res.append([datetime_str, 3, 2, values_128[2], topology_index])
        res.append([datetime_str, 3, 34, values_128[3], topology_index])


df = pd.DataFrame(res)
df.columns = ["Timestamp", "AssetID", "MeasurementID", "Value", "Topology"]

df.to_csv("themis_results.csv")