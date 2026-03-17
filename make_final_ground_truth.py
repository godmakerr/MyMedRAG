import pandas as pd
import json
import os

test_csv_path = './dataset/ddxplus_data/release_test_patients/release_test_patients'
conditions_json_path = './dataset/ddxplus_data/release_conditions.json'
output_csv = './dataset/AI Data Set with Categories.csv'


disease_taxonomy = {
    "acute_copd_exacerbation_infection": ("Respiratory System", "COPD"),
    "bronchiectasis": ("Respiratory System", "Bronchial Disease"),
    "bronchiolitis": ("Respiratory System", "Infection"),
    "acute_bronchitis": ("Respiratory System", "Infection"),
    "pneumonia": ("Respiratory System", "Infection"),
    "urti": ("Respiratory System", "Upper Respiratory"),
    "influenza": ("Respiratory System", "Viral Infection"),
    "asthma_exacerbation": ("Respiratory System", "Asthma"),
    "chronic_rhinosinusitis": ("Respiratory System", "Sinusitis"),
    "whooping_cough": ("Respiratory System", "Infection"),
    
    "acute_pulmonary_edema": ("Cardiovascular System", "Heart Failure"),
    "stable_angina": ("Cardiovascular System", "Ischemic Heart Disease"),
    "unstable_angina": ("Cardiovascular System", "Ischemic Heart Disease"),
    "myocarditis": ("Cardiovascular System", "Inflammation"),
    "pericarditis": ("Cardiovascular System", "Inflammation"),
    "chf": ("Cardiovascular System", "Heart Failure"),
    "psvt": ("Cardiovascular System", "Arrhythmia"),
    "atrial_fibrillation": ("Cardiovascular System", "Arrhythmia"),
    
    "gerd": ("Gastrointestinal System", "Esophageal"),
    "acute_pancreatitis": ("Gastrointestinal System", "Pancreas"),
    "chronic_pancreatitis": ("Gastrointestinal System", "Pancreas"),
    "appendicitis": ("Gastrointestinal System", "Acute Abdomen"),
    "cholecystitis": ("Gastrointestinal System", "Biliary"),
    "inguinal_hernia": ("Gastrointestinal System", "Hernia"),
    "pancreatic_neoplasm": ("Gastrointestinal System", "Neoplasm"),

    "anemia": ("Hematologic System", "Anemia"),
    "spontaneous_pneumothorax": ("Respiratory System", "Pleural Disease"),
    "boerhaave": ("Gastrointestinal System", "Emergency"),
    "slep": ("Immune System", "Autoimmune"),
    "hiv_prim": ("Immune System", "Infection"),
    "sle": ("Immune System", "Autoimmune"),
    "anaphylaxis": ("Immune System", "Allergy"),
    "tuberculosis": ("Respiratory System", "Infection"),
    "sarcoidosis": ("Multi-system", "Inflammation"),

    "default": ("General System", "General Disease")
}

print("Ground Truth...")

try:
    df = pd.read_csv(test_csv_path)
    print(f"{len(df)}")

    gt_df = pd.DataFrame()
    
    if 'PATIENT_ID' in df.columns:
        gt_df['Participant No.'] = df['PATIENT_ID']
    else:
        gt_df['Participant No.'] = [f"test_{i+1}" for i in range(len(df))]

    gt_df['Processed Diagnosis'] = df['PATHOLOGY']
    gt_df['Diagnoses (related to pain)'] = df['PATHOLOGY']
    
    print("Level 1 / Level 2...")
    
    l1_list = []
    l2_list = []
    
    for pathology in df['PATHOLOGY']:
        if pathology in disease_taxonomy:
            l1, l2 = disease_taxonomy[pathology]
        else:
            if 'card' in pathology or 'heart' in pathology:
                l1, l2 = ("Cardiovascular System", "Heart Disease")
            elif 'pne' in pathology or 'bronch' in pathology:
                l1, l2 = ("Respiratory System", "Lung Disease")
            else:
                l1, l2 = disease_taxonomy["default"]
                
        l1_list.append(l1)
        l2_list.append(l2)
        
    gt_df['Level 1'] = l1_list
    gt_df['Level 2'] = l2_list

    gt_df.to_csv(output_csv, index=False)
    print(f"{output_csv}")
    print(gt_df[['Processed Diagnosis', 'Level 1', 'Level 2']].head())

except Exception as e:
    print(f"{e}")