import pandas as pd
import json

df = pd.read_csv('M250_S139_S200_separate_dataset_4000_samples.csv')
sample_counts = df.groupby('Sample_ID').size()

nunique_per_sample = df.groupby('Sample_ID')[['Specimen_Type', 'Repair_Location']].nunique()
first_sample = df[df['Sample_ID'] == 0]

with open('check_data.txt', 'w') as f:
    f.write(f"Total Samples: {len(sample_counts)}\n")
    f.write(f"Conditions: {df['Condition'].unique().tolist()}\n")
    f.write(f"Repair_Location unique values: {df['Repair_Location'].unique().tolist()}\n")
    f.write(f"\nIs Specimen_Type constant per Sample? {(nunique_per_sample['Specimen_Type'] == 1).all()}\n")
    f.write(f"Is Repair_Location constant per Sample? {(nunique_per_sample['Repair_Location'] == 1).all()}\n")
    f.write("\nFirst Sample structure:\n")
    f.write(first_sample.to_string())


