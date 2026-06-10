import pandas as pd

df = pd.read_csv('M250_S139_S200_separate_dataset_4000_samples.csv')
pvt = df.pivot(index='Sample_ID', columns='Condition', values=['UTS', 'YS', 'K1c', 'Elongation', 'HAZ_Width', 'Gauge_Depth', 'Austenite_Content'])

# Flatten column names e.g., 'UTS', 'R0' -> 'R0_UTS'
pvt.columns = [f"{col[1]}_{col[0]}" for col in pvt.columns]

# Get constants
constants = df[['Sample_ID', 'Specimen_Type', 'Repair_Location']].drop_duplicates().set_index('Sample_ID')
final_df = pvt.join(constants)

# One-hot encoding
final_df = pd.get_dummies(final_df, columns=['Specimen_Type', 'Repair_Location'])

with open('cols_pivoted.txt', 'w') as f:
    f.write(str(final_df.shape) + "\n")
    f.write(", ".join(final_df.columns.tolist()))

