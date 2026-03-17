import pandas as pd
import json
import ast
from tqdm import tqdm
from config_01 import TRAIN_DATA_PATH, SYMPTOM_STATS_PATH

def parse_evidence(ev_str):
    """
    Robust parsing function to handle various string formats.
    """
    if pd.isna(ev_str):
        return []
    try:
        # Case 1: Standard list string "['E_1', 'E_2']"
        return ast.literal_eval(ev_str)
    except:
        try:
            # Case 2: Malformed list without quotes "[E_1, E_2]"
            clean_str = ev_str.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
            return [x.strip() for x in clean_str.split(',') if x.strip()]
        except Exception as e:
            # Return empty list on failure
            return []

def main():
    print(f"Reading training data from: {TRAIN_DATA_PATH} ...")
    
    # Try reading with 'EVIDENCES' column first, fallback to 'EVIDENCE'
    try:
        df = pd.read_csv(TRAIN_DATA_PATH, usecols=['PATHOLOGY', 'EVIDENCES'])
    except ValueError:
        print("Column 'EVIDENCES' not found. Trying 'EVIDENCE'...")
        df = pd.read_csv(TRAIN_DATA_PATH, usecols=['PATHOLOGY', 'EVIDENCE'])
        df.rename(columns={'EVIDENCE': 'EVIDENCES'}, inplace=True)

    print(f"Successfully loaded {len(df)} records. Analyzing symptom frequencies...")

    stats = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Normalize key by stripping whitespace
        disease = row['PATHOLOGY'].strip() 
        evidence_list = parse_evidence(row['EVIDENCES'])
        
        if not evidence_list:
            continue

        if disease not in stats:
            stats[disease] = {'total_cases': 0, 'symptoms': {}}
        
        stats[disease]['total_cases'] += 1
        
        for ev in evidence_list:
            if ev not in stats[disease]['symptoms']:
                stats[disease]['symptoms'][ev] = 0
            stats[disease]['symptoms'][ev] += 1

    # Calculate probabilities and filter
    final_probs = {}
    kept_symptoms_count = 0
    
    print("Calculating probabilities and filtering rare symptoms (Threshold > 1%)...")
    
    for disease, data in stats.items():
        total = data['total_cases']
        final_probs[disease] = {}
        
        for symptom, count in data['symptoms'].items():
            prob = count / total
            # Lowered threshold to 0.01 (1%) to ensure connectivity
            if prob > 0.01: 
                final_probs[disease][symptom] = prob
                kept_symptoms_count += 1

    print(f"Statistics extraction complete!")
    print(f"Total diseases processed: {len(final_probs)}")
    print(f"Total symptom edges preserved: {kept_symptoms_count}")
    
    # Sample check for Pneumonia
    for k in final_probs:
        if "pneumonia" in k.lower():
            print(f"Sample Check - {k}: Contains {len(final_probs[k])} associated symptoms.")

    with open(SYMPTOM_STATS_PATH, 'w') as f:
        json.dump(final_probs, f, indent=4)
    print(f"Stats saved to: {SYMPTOM_STATS_PATH}")

if __name__ == "__main__":
    main()