import os
import pandas as pd

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'M250_S139_S200_separate_dataset_4000_samples.csv')
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        pvt = df.pivot(index='Sample_ID', columns='Condition', values=['UTS', 'YS', 'K1c', 'Elongation', 'HAZ_Width', 'Gauge_Depth', 'Austenite_Content'])
        pvt.columns = [f"{col[1]}_{col[0]}" for col in pvt.columns]
        constants = df[['Sample_ID', 'Specimen_Type', 'Repair_Location']].drop_duplicates().set_index('Sample_ID')
        final_df = pvt.join(constants)
        final_df = pd.get_dummies(final_df, columns=['Specimen_Type', 'Repair_Location'])
        with open('cols_pivoted.txt', 'w') as f:
            f.write(str(final_df.shape) + "\n")
            f.write(", ".join(final_df.columns.tolist()))

