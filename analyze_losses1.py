import pandas as pd

BASELINE_CSV = "./test_results_topk1_topn1_matchn5_1000_1051_MedRAG.csv"
HYBRID_CSV = "./results_Hybrid_KARE_1000_1051.csv"

def clean_id(val):
    return str(val).lower().strip().replace("test_", "").replace("participant_", "").replace(".json", "")

def normalize(text):
    if pd.isna(text): return ""
    return str(text).lower().strip().replace("**diagnosis**:", "")

def main():
    base = pd.read_csv(BASELINE_CSV)
    hybrid = pd.read_csv(HYBRID_CSV)
    
    base['clean_id'] = base['Participant No.'].apply(clean_id)
    hybrid['clean_id'] = hybrid['Participant No.'].apply(clean_id)
    
    merged = pd.merge(base, hybrid, on='clean_id', suffixes=('_base', '_kare'))
    
    print("---  分析 KARE 错误案例 (Baseline对 KARE错) ---")
    
    for _, row in merged.iterrows():
        truth = normalize(row['True Diagnosis_base'])
        pred_base = normalize(row['Generated Diagnosis_base'])
        pred_kare = normalize(row['Generated Diagnosis_kare'])
        
        base_correct = truth in pred_base or pred_base in truth
        kare_correct = truth in pred_kare or pred_kare in truth
        
        if base_correct and not kare_correct:
            print(f"\n[Patient {row['clean_id']}]")
            print(f"  真实病因: {row['True Diagnosis_base']}")
            print(f"  Baseline: {row['Generated Diagnosis_base']} (True)")
            print(f"  KARE预测: {row['Generated Diagnosis_kare']} (False)")

if __name__ == "__main__":
    main()