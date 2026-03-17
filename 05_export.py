import networkx as nx
import pickle
import pandas as pd
from config_01 import (
    AUGMENTED_GRAPH_PATH, FINAL_KG_XLSX, 
    FINAL_GT_CSV, TEST_DATA_PATH, DISEASE_TAXONOMY
)

def export_kg_to_excel(G):
    print("Exporting Graph to Excel...")
    data = []
    
    for u, v, attr in G.edges(data=True):
        relation = attr.get('relation', 'related_to')
        
        # Determine object content based on node type
        # For symptom nodes or hierarchy, the node ID is the content
        # For difference nodes, the 'content' attribute is the text
        if G.nodes[v].get('type') == 'diagnostic_difference':
            obj_text = G.nodes[v].get('content', str(v))
        else:
            obj_text = str(v)

        data.append({
            'subject': u,
            'relation': relation,
            'object': obj_text
        })
    
    df = pd.DataFrame(data)
    # Ensure columns order
    df = df[['subject', 'relation', 'object']]
    
    df.to_excel(FINAL_KG_XLSX, index=False)
    print(f"Knowledge Graph saved to {FINAL_KG_XLSX}")

def generate_ground_truth_csv():
    print("Generating Ground Truth CSV...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    gt_data = []
    
    for i, row in df_test.iterrows():
        # Handle ID (Use integer or string depending on main.py requirement)
        # Using string 'test_X' based on previous context
        patient_id = row.get('PATIENT_ID', f"test_{i+1}")
        pathology = row['PATHOLOGY']
        
        # Get Levels
        if pathology in DISEASE_TAXONOMY:
            l1, l2 = DISEASE_TAXONOMY[pathology]
        else:
            l1, l2 = DISEASE_TAXONOMY["default"]
            
        gt_data.append({
            'Participant No.': patient_id,
            'Processed Diagnosis': pathology,
            'Diagnoses (related to pain)': pathology, # Placeholder
            'Level 1': l1,
            'Level 2': l2
        })
        
    df_gt = pd.DataFrame(gt_data)
    df_gt.to_csv(FINAL_GT_CSV, index=False)
    print(f"Ground Truth CSV saved to {FINAL_GT_CSV}")

def main():
    # 1. Export Graph
    # If Step 4 was skipped, load base_graph.gpickle instead
    try:
        with open(AUGMENTED_GRAPH_PATH, 'rb') as f:
            G = pickle.load(f)
    except FileNotFoundError:
        print("Augmented graph not found, falling back to base graph...")
        from _01_config import GRAPH_PICKLE_PATH
        with open(GRAPH_PICKLE_PATH, 'rb') as f:
            G = pickle.load(f)
            
    export_kg_to_excel(G)
    
    # 2. Generate Ground Truth
    generate_ground_truth_csv()

if __name__ == "__main__":
    main()