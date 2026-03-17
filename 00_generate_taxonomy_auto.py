import json
import os
import numpy as np
import openai
import faiss  # <--- 用 Faiss 替代 sklearn
# from sklearn.cluster import KMeans # <--- 彻底删掉这一行，解决报错根源

from config_01 import (
    SYMPTOM_STATS_PATH, API_KEY, BASE_URL
)

# Initialize OpenAI
client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_embedding(text):
    """
    Get vector representation using OpenAI Embeddings.
    """
    text = text.replace("_", " ")
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error for {text}: {e}")
        return [0.0] * 1536

def generate_cluster_name(disease_list, level_name):
    """
    Ask LLM to name a cluster.
    """
    diseases_str = ", ".join(disease_list)
    prompt = (
        f"Here is a list of medical diseases: [{diseases_str}]. "
        f"They have been clustered together based on semantic similarity. "
        f"Please provide a short, precise medical category name (Title Case) for this group. "
        f"This will be used as a '{level_name}' in a knowledge graph. "
        f"Return ONLY the category name, nothing else."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip().replace(".", "")
    except Exception as e:
        print(f"LLM Naming error: {e}")
        return "Unknown Category"

def perform_clustering(embeddings, n_clusters):
    """
    使用 Faiss 进行 KMeans 聚类，替代 sklearn
    """
    X = np.array(embeddings).astype('float32')
    # 确保内存连续 (Faiss 要求)
    X = np.ascontiguousarray(X)
    
    d = X.shape[1]
    
    # 初始化 Faiss KMeans
    # niter: 迭代次数, verbose: 是否打印进度
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True, seed=42)
    
    # 训练
    kmeans.train(X)
    
    # 分配标签 (也就是 sklearn 的 predict)
    # search 返回 (距离, 索引/标签)
    _, labels = kmeans.index.search(X, 1)
    
    # 展平为一维数组
    return labels.flatten()

def main():
    print("--- Starting Automatic Taxonomy Generation (Faiss Version) ---")
    
    # 1. Load Diseases
    if not os.path.exists(SYMPTOM_STATS_PATH):
        print(f"Error: {SYMPTOM_STATS_PATH} not found. Run 02_extract_stats.py first.")
        return

    with open(SYMPTOM_STATS_PATH, 'r') as f:
        data = json.load(f)
    
    diseases = sorted(list(data.keys()))
    print(f"Loaded {len(diseases)} diseases.")

    # 2. Generate Embeddings
    print("Generating embeddings for diseases...")
    embeddings = []
    for d in diseases:
        emb = get_embedding(d)
        embeddings.append(emb)
    
    # 3. Clustering Tier 2 (Subcategories)
    n_clusters_l2 = max(2, len(diseases) // 4) 
    print(f"Clustering into {n_clusters_l2} Subcategories (Level 2) using Faiss...")
    
    # --- 替换点 1 ---
    labels_l2 = perform_clustering(embeddings, n_clusters_l2)

    cluster_map_l2 = {}
    for idx, label in enumerate(labels_l2):
        label = int(label) # numpy int 转 python int
        if label not in cluster_map_l2:
            cluster_map_l2[label] = []
        cluster_map_l2[label].append(diseases[idx])

    # 4. Name Tier 2 Clusters via LLM
    print("Naming Tier 2 clusters via LLM...")
    d_to_l2 = {} 
    l2_to_diseases = {}

    for label, d_list in cluster_map_l2.items():
        l2_name = generate_cluster_name(d_list, "Medical Subcategory")
        print(f"  Cluster {label}: {l2_name} -> {d_list}")
        l2_to_diseases[l2_name] = d_list
        for d in d_list:
            d_to_l2[d] = l2_name

    # 5. Clustering Tier 1 (Systems)
    unique_l2_names = list(l2_to_diseases.keys())
    n_clusters_l1 = max(2, len(unique_l2_names) // 3)
    print(f"\nClustering {len(unique_l2_names)} Subcategories into {n_clusters_l1} Systems (Level 1)...")
    
    l2_embeddings = []
    for l2 in unique_l2_names:
        l2_embeddings.append(get_embedding(l2))
    
    # --- 替换点 2 ---
    labels_l1 = perform_clustering(l2_embeddings, n_clusters_l1)
    
    cluster_map_l1 = {}
    for idx, label in enumerate(labels_l1):
        label = int(label)
        if label not in cluster_map_l1:
            cluster_map_l1[label] = []
        cluster_map_l1[label].append(unique_l2_names[idx])
        
    # 6. Name Tier 1 Clusters via LLM
    print("Naming Tier 1 clusters via LLM...")
    l2_to_l1 = {}
    
    for label, l2_list in cluster_map_l1.items():
        l1_name = generate_cluster_name(l2_list, "Body System / Medical Specialty")
        print(f"  System {label}: {l1_name} -> {l2_list}")
        for l2 in l2_list:
            l2_to_l1[l2] = l1_name

    # 7. Generate Final Dictionary Code
    print("\n--- Generation Complete. Generating Python Code ---")
    print("\n# Copy this into config_01.py to replace DISEASE_TAXONOMY")
    print("DISEASE_TAXONOMY = {")
    
    sorted_diseases = sorted(diseases)
    for d in sorted_diseases:
        l2 = d_to_l2.get(d, "Unclassified")
        l1 = l2_to_l1.get(l2, "Unclassified")
        print(f'    "{d}": ("{l1}", "{l2}"),')
        
    print("}")

if __name__ == "__main__":
    main()