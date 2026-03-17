# --- START OF FILE 07_build_kare_index.py ---

import networkx as nx
import pickle
import numpy as np
import json
import os
from cdlib import algorithms
from tqdm import tqdm
import openai

# 复用你之前的配置
from config_01 import (
    AUGMENTED_GRAPH_PATH, 
    API_KEY, BASE_URL
)

# 定义新的保存路径
KARE_INDEX_PATH = "data/kare_community_index.pkl"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_embedding(text):
    """复用 embedding 函数"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 1536

def describe_community(G, community_nodes):
    """
    将社区转化为一段文本描述，用于 Embedding。
    策略：提取社区内的所有疾病名和高频症状。
    """
    subgraph = G.subgraph(community_nodes)
    
    # 提取节点信息
    diseases = [n for n in subgraph.nodes() if subgraph.nodes[n].get('type') == 'disease']
    symptoms = [n for n in subgraph.nodes() if subgraph.nodes[n].get('type') == 'symptom']
    
    # 如果有鉴别诊断节点（MedRAG特有），也提取出来
    diffs = [n for n in subgraph.nodes() if subgraph.nodes[n].get('type') == 'diagnostic_difference']
    
    # 构造描述文本
    # 格式: "Diseases: [A, B]. Key Symptoms: [X, Y]. Differences: [Info...]"
    desc_text = f"Diseases Cluster: {', '.join(diseases[:10])}. "
    desc_text += f"Associated Symptoms: {', '.join(symptoms[:15])}. "
    
    if diffs:
        # 简单拼接一个 difference 的摘要，避免 token 过长
        desc_text += "Contains diagnostic comparisons."
        
    return desc_text

def main():
    print(f"Loading augmented graph from {AUGMENTED_GRAPH_PATH}...")
    with open(AUGMENTED_GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)
    
    # --- 1. KARE 核心步骤：社区检测 (Community Detection) ---
    print("Running Leiden Community Detection (KARE Algorithm)...")
    # 将有向图转为无向图用于聚类
    G_undirected = G.to_undirected()
    
    # 使用 Leiden 算法 (比 Louvain 更稳健，SOTA标配)
    coms = algorithms.leiden(G_undirected)
    
    print(f"Detected {len(coms.communities)} semantic communities.")
    
    # --- 2. 构建社区索引 ---
    community_index = []
    
    print("Generating embeddings for each community...")
    for cid, nodes in tqdm(enumerate(coms.communities), total=len(coms.communities)):
        # 忽略太小的社区（通常是噪声）
        if len(nodes) < 3:
            continue
            
        # 生成社区的文本描述
        description = describe_community(G, nodes)
        
        # 计算社区向量
        vector = get_embedding(description)
        
        community_index.append({
            "id": cid,
            "nodes": nodes,
            "description": description,
            "vector": vector
        })
        
    print(f"Index built. Valid communities: {len(community_index)}")
    
    # --- 3. 保存索引 ---
    os.makedirs(os.path.dirname(KARE_INDEX_PATH), exist_ok=True)
    with open(KARE_INDEX_PATH, 'wb') as f:
        pickle.dump(community_index, f)
    
    print(f"KARE Index saved to {KARE_INDEX_PATH}")

if __name__ == "__main__":
    main()