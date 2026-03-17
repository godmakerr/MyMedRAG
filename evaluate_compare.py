import pandas as pd

BASELINE_CSV = "./test_results_topk1_topn0_matchn5_1000_1051_MedRAG.csv" 
HYBRID_CSV = "./results_Hybrid_KARE_1000_1051.csv"  

def clean_id(val):
    return str(val).lower().strip().replace("test_", "").replace("participant_", "").replace(".json", "")

def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text).lower().strip().rstrip(".")
    text = text.replace("**diagnosis**:", "").replace("diagnosis:", "")
    return text.strip()

def calculate_accuracy(df, name):
    df['clean_pred'] = df['Generated Diagnosis'].apply(normalize_text)
    df['clean_true'] = df['True Diagnosis'].apply(normalize_text)
    
    correct = 0
    for _, row in df.iterrows():
        if row['clean_true'] in row['clean_pred'] or row['clean_pred'] in row['clean_true']:
            correct += 1
            
    acc = (correct / len(df)) * 100
    print(f"[{name}] Accuracy: {acc:.2f}% ({correct}/{len(df)})")
    return df

def main():
    try:
        df_base = pd.read_csv(BASELINE_CSV)
        df_hybrid = pd.read_csv(HYBRID_CSV)
    except Exception as e:
        print(f"{e}")
        return

    df_base['clean_id'] = df_base['Participant No.'].apply(clean_id)
    df_hybrid['clean_id'] = df_hybrid['Participant No.'].apply(clean_id)

    common_ids = set(df_base['clean_id']).intersection(set(df_hybrid['clean_id']))
    print(f"Total: {len(common_ids)} ")

    df_base = df_base[df_base['clean_id'].isin(common_ids)].sort_values('clean_id')
    df_hybrid = df_hybrid[df_hybrid['clean_id'].isin(common_ids)].sort_values('clean_id')

    calculate_accuracy(df_base, "Baseline")
    calculate_accuracy(df_hybrid, "Hybrid KARE")

if __name__ == "__main__":
    main()