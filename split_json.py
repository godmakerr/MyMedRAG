import pandas as pd
import json
import os
import ast

source_file = './dataset/ddxplus_data/release_train_patients/release_train_patients'

output_folder = './dataset/df/train'
os.makedirs(output_folder, exist_ok=True)

print(f"{source_file}")

try:
    df = pd.read_csv(source_file)
    print(f" {len(df)}")

    for i, row in df.iterrows():
        patient_data = row.to_dict()
        
        if 'PATIENT_ID' in patient_data:
            patient_data['Participant No.'] = patient_data['PATIENT_ID']
        else:
            patient_data['Participant No.'] = f"train_{i+1}"

        keys_to_parse = ['EVIDENCE', 'INITIAL_EVIDENCE', 'symptoms']
        for key in keys_to_parse:
            if key in patient_data and isinstance(patient_data[key], str):
                try:
                    patient_data[key] = ast.literal_eval(patient_data[key])
                except:
                    pass 

        if 'EVIDENCE' in patient_data:
            patient_data['Symptoms'] = patient_data['EVIDENCE']
            
        if 'PATHOLOGY' in patient_data:
            patient_data['Processed Diagnosis'] = patient_data['PATHOLOGY']

        file_name = f"participant_{i+1}.json"
        save_path = os.path.join(output_folder, file_name)

        def convert_numpy(obj):
            if isinstance(obj, (int, float)):
                return obj
            raise TypeError

        with open(save_path, 'w', encoding='utf-8') as out_f:
            json.dump(patient_data, out_f, indent=4, default=str)

    print(f"{len(df)} ")
    print(f"{output_folder}")
    print(f"range:range(1, {len(df) + 1})")

except Exception as e:
    print(f"{e}")