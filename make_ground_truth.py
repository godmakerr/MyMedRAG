import pandas as pd
import os

source_file = './dataset/ddxplus_data/release_test_patients/release_test_patients'

output_file = './dataset/AI Data Set with Categories.csv'

print(f"reading: {source_file}")

try:

    df = pd.read_csv(source_file)
    
    gt_df = pd.DataFrame()

    if 'PATIENT_ID' in df.columns:
        gt_df['Participant No.'] = df['PATIENT_ID']
    else:
        gt_df['Participant No.'] = [f"test_{i+1}" for i in range(len(df))]

    gt_df['Processed Diagnosis'] = df['PATHOLOGY']

    gt_df['Diagnoses (related to pain)'] = df['PATHOLOGY']

    # defult setting
    gt_df['Level 1'] = 'General System'
    gt_df['Level 2'] = 'General Disease'

    gt_df.to_csv(output_file, index=False)

    print(f"{output_file}")
    print(f"{list(gt_df.columns)}")
    print(f"{len(gt_df)}")
    print("-" * 30)

except Exception as e:
    print(f"{e}")