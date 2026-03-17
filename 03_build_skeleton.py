import networkx as nx
import json
import pickle
import difflib 
from config_01 import (
    DISEASE_TAXONOMY, SYMPTOM_STATS_PATH, 
    GRAPH_PICKLE_PATH
)

def normalize_key(text):
    return text.lower().replace("_", " ").replace("-", " ").strip()

def main():
    print("Initializing Graph...")
    G = nx.DiGraph()
    root_node = "Global_Root"
    G.add_node(root_node, type="root", label="Medical Diagnosis")

    # Load stats
    print(f"Loading stats from {SYMPTOM_STATS_PATH}...")
    with open(SYMPTOM_STATS_PATH, 'r') as f:
        disease_symptoms = json.load(f)
    
    stats_lookup = {}
    for k in disease_symptoms.keys():
        clean_k = normalize_key(k)
        stats_lookup[clean_k] = k 

    print(f"Loaded {len(stats_lookup)} diseases from stats.")
    print("Building Graph Skeleton (Taxonomy + Diseases + Symptoms)...")
    
    match_count = 0
    fail_count = 0

    for disease_name, (system, subcat) in DISEASE_TAXONOMY.items():
        if disease_name == "default": continue

        if not G.has_node(system):
            G.add_node(system, type="system", level=1)
            G.add_edge(root_node, system, relation="has_category")
        
        if not G.has_node(subcat):
            G.add_node(subcat, type="subcategory", level=2)
            G.add_edge(system, subcat, relation="has_subcategory")

        G.add_node(disease_name, type="disease", level=3)
        G.add_edge(subcat, disease_name, relation="has_disease")

        target_key = normalize_key(disease_name)
        
        if target_key in stats_lookup:
            original_key = stats_lookup[target_key]
            symptoms_data = disease_symptoms[original_key]
            for symptom_code, prob in symptoms_data.items():
                if not G.has_node(symptom_code):
                    G.add_node(symptom_code, type="symptom", level=4)
                G.add_edge(disease_name, symptom_code, relation="has_symptom", weight=prob)
            match_count += 1
            
        else:
            all_keys = list(stats_lookup.keys())
            matches = difflib.get_close_matches(target_key, all_keys, n=1, cutoff=0.6)
            
            if matches:
                guessed_key = matches[0]
                original_key = stats_lookup[guessed_key]
                print(f"?? Fuzzy Match: Config '{disease_name}' -> Data '{original_key}'")
                
                symptoms_data = disease_symptoms[original_key]
                for symptom_code, prob in symptoms_data.items():
                    if not G.has_node(symptom_code):
                        G.add_node(symptom_code, type="symptom", level=4)
                    G.add_edge(disease_name, symptom_code, relation="has_symptom", weight=prob)
                match_count += 1
            else:
                print(f"?? Warning: No match found for '{disease_name}' (Normalized: '{target_key}')")
                fail_count += 1

    print("-" * 30)
    print(f"Graph construction complete.")
    print(f"Successful matches: {match_count}")
    print(f"Failed matches: {fail_count}")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    
    if G.has_node("Acute bronchitis"):
         sym_count = len([n for n in G.neighbors("Acute bronchitis") if G.nodes[n].get('type') == 'symptom'])
         print(f"Verification: 'Acute bronchitis' is now connected to {sym_count} symptoms.")

    print(f"Saving graph to {GRAPH_PICKLE_PATH}...")
    with open(GRAPH_PICKLE_PATH, 'wb') as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    main()