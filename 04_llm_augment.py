import networkx as nx
import pickle
import openai
from tqdm import tqdm
from config_01 import (
    GRAPH_PICKLE_PATH, AUGMENTED_GRAPH_PATH, 
    API_KEY, BASE_URL
)

# Initialize OpenAI Client
client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_diagnostic_difference(disease_a, disease_b):
    """
    Generates differentiation knowledge between two similar diseases.
    """
    prompt = (
        f"Identify the key diagnostic differences between {disease_a} and {disease_b}. "
        f"Focus on clinical presentation, key symptoms, and history. "
        f"Provide a concise summary."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "Comparison unavailable due to API error."

def main():
    print("Loading skeleton graph...")
    with open(GRAPH_PICKLE_PATH, 'rb') as f:
        G = pickle.load(f)

    # Find all Tier 2 nodes (Subcategories)
    subcategories = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'subcategory']
    
    print(f"Found {len(subcategories)} subcategories for augmentation.")

    # Counter to limit costs during testing
    api_call_count = 0
    MAX_API_CALLS = -1  # Set to -1 to run for ALL diseases (costs money!)

    for subcat in tqdm(subcategories):
        # Get all diseases under this subcategory (Siblings)
        siblings = [n for n in G.successors(subcat) if G.nodes[n].get('type') == 'disease']
        
        # Compare pairs of siblings
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                
                if 0 < MAX_API_CALLS <= api_call_count:
                    print("API call limit reached. Stopping augmentation.")
                    break
                
                d1 = siblings[i]
                d2 = siblings[j]
                
                # Generate Knowledge
                diff_text = get_diagnostic_difference(d1, d2)
                api_call_count += 1
                
                # Create a specific node for this difference
                diff_node_id = f"Diff_{d1}_vs_{d2}"
                
                # Add node to graph
                G.add_node(diff_node_id, type="diagnostic_difference", content=diff_text)
                
                # Link both diseases to this difference node
                # The relation logic: d1 is differentiated from d2 by this node
                G.add_edge(d1, diff_node_id, relation="differentiated_by")
                G.add_edge(d2, diff_node_id, relation="differentiated_by")
            
            if 0 < MAX_API_CALLS <= api_call_count:
                break
        
        if 0 < MAX_API_CALLS <= api_call_count:
            break

    print(f"Augmentation finished. Total API calls: {api_call_count}")
    print(f"Saving augmented graph to {AUGMENTED_GRAPH_PATH}...")
    with open(AUGMENTED_GRAPH_PATH, 'wb') as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    main()