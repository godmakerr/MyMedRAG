# --- START OF FILE main_Hybrid_KARE1.py ---

# 1. 必须最先导入 main_MedRAG
from main_MedRAG import (
    get_query_embedding, Faiss, document_embeddings, documents, 
    client, InferenceClient, hf_token,
    get_system_prompt_for_RAGKG, get_additional_info_from_level_2
)

# 2. 再导入 authentication
from authentication import (
    test_folder_path, 
    ground_truth_file_path, 
    augmented_features_path 
)

# 3. 其他库
import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import glob

# --- KARE 配置 ---
KARE_INDEX_PATH = "data/kare_community_index.pkl"

print("Loading KARE Index...")
if os.path.exists(KARE_INDEX_PATH):
    with open(KARE_INDEX_PATH, 'rb') as f:
        kare_index = pickle.load(f)
    # 【关键】强制转为 float32 且内存连续
    kare_vectors = np.array([item['vector'] for item in kare_index], dtype='float32')
    print(f"Loaded {len(kare_index)} KARE communities.")
else:
    print("WARNING: KARE Index not found. Please run 07_build_kare_index.py first!")
    kare_index = []
    kare_vectors = np.array([])

def retrieve_kare_context(query_emb, top_k=2):
    if len(kare_index) == 0:
        return ""
    
    # 确保 query 也是 float32 且连续
    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32).reshape(1, -1)
    
    sims = cosine_similarity(query_emb, kare_vectors)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    
    context_parts = []
    for idx in top_indices:
        com = kare_index[idx]
        desc = com['description']
        context_parts.append(f"[Medical Knowledge Cluster {idx}]: {desc}")
    
    return "\n".join(context_parts)

def generate_diagnosis_report_hybrid(path, query, retrieved_documents, kare_context, i, top_n, match_n, model):
    system_prompt = get_system_prompt_for_RAGKG()
    
    try:
        additional_info = get_additional_info_from_level_2(i, path, top_n=top_n, match_n=match_n)
    except Exception as e:
        # print(f"Warning: Basic KG info failed: {e}")
        additional_info = "Not available."
        
    if additional_info is None: additional_info = "No basic KG info."

    prompt = f"""
Patient Case Query:
{query}

--- Source A: Similar Past Patient Cases (Experience) ---
{retrieved_documents}

--- Source B: KARE Knowledge Graph Communities (Medical Evidence) ---
The following are retrieved clusters of diseases and differential diagnoses based on semantic graph analysis:
{kare_context}

--- Source C: Basic Category Info ---
{additional_info}

--- Instructions ---
Refer to the "KARE Knowledge Graph Communities" to distinguish between similar diseases (e.g., if symptoms match multiple diseases, use the clusters to differentiate).
Combine this with the "Similar Past Patient Cases" to make your final decision.
Now complete the tasks in the required format.
"""

    if model in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-0125']:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        full_prompt = f"<s>[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]"
        LLMclient = InferenceClient("meta-llama/Meta-Llama-3.1-8B-Instruct", token=hf_token)
        response = LLMclient.text_generation(prompt=full_prompt, max_new_tokens=400)
        return response

def main():
    # 1. 加载标准答案
    ground_truth = pd.read_csv(ground_truth_file_path, header=0)
    results = []
    
    # --- 配置区域 ---
    topk_faiss = 1   # 基础 RAG 检索数量
    topk_kare = 2    # KARE 社区检索数量
    my_model = "gpt-4o"
    
    # 【关键修改】使用 range 控制测试范围 (与 main.py 保持 100% 一致)
    # 例如：测 ID 1000 到 1050 (不包含1051)
    target_range = range(1000, 1051) 
    
    # 【关键修改】动态生成输出文件名
    # 结果将保存为: ./test_results_Hybrid_KARE_1000_1051.csv
    output_filename = f"./test_results_Hybrid_KARE_{target_range.start}_{target_range.stop}.csv"
    
    print(f"Running Hybrid RAG on range: {target_range}")
    print(f"Output will be saved to: {output_filename}")

    # --- 2. 修复知识库向量 (防报错核心) ---
    global document_embeddings
    print("Preparing Document Embeddings...")
    # 强制转为 Numpy 且 Float32
    if not isinstance(document_embeddings, np.ndarray):
        document_embeddings = np.array(document_embeddings)
    
    # 强制内存连续 (C-Contiguous)，这是解决 "input not a numpy array" 的终极方案
    document_embeddings = np.ascontiguousarray(document_embeddings, dtype=np.float32)
        
    print(f"Doc Embeddings Ready: Shape={document_embeddings.shape}, C-Contiguous={document_embeddings.flags['C_CONTIGUOUS']}")

    # --- 3. 开始循环 ---
    for i in tqdm(target_range):
        # 手动拼接文件名，确保只测指定 ID
        filename = f"participant_{i}.json"
        file_path = os.path.join(test_folder_path, filename)

        # 文件检查
        if not os.path.exists(file_path):
            # print(f"Skipping {i} (File not found)") # 可选：打印跳过信息
            continue

        try:
            with open(file_path, 'r') as file:
                new_patient_case = json.load(file)
            
            # 获取 ID 和 查询文本
            participant_no = new_patient_case.get('Participant No.', f"participant_{i}")
            query_text = json.dumps(new_patient_case)

            # --- A. 获取 Query 向量并修复格式 ---
            raw_emb = get_query_embedding(query_text)
            # 同样强制连续化
            query_embedding = np.ascontiguousarray(raw_emb, dtype=np.float32)
            # 确保是二维 (1, dim)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # --- B. 检索相似病人 (Faiss) ---
            # 此时传入的两个参数都是完美的 float32 连续数组，绝对不会报错
            indices = Faiss(document_embeddings, query_embedding, k=topk_faiss)
            
            retrieved_docs_content = []
            if topk_faiss > 0:
                indices = Faiss(document_embeddings, query_embedding, k=topk_faiss)
                for idx in indices[0]:
                    with open(documents[idx], 'r') as f:
                        retrieved_docs_content.append(json.load(f))
            else:
                pass
            
            # --- C. 检索 KARE 知识社区 ---
            kare_context_text = retrieve_kare_context(query_embedding, top_k=topk_kare)

            # --- D. 获取真实诊断 (Ground Truth) ---
            # 尝试多种 ID 格式匹配
            true_rows = ground_truth.loc[ground_truth['Participant No.'] == participant_no]
            if true_rows.empty:
                true_rows = ground_truth.loc[ground_truth['Participant No.'] == str(participant_no)]
            if true_rows.empty:
                true_rows = ground_truth.loc[ground_truth['Participant No.'] == f"participant_{i}"]
            if true_rows.empty:
                true_rows = ground_truth.loc[ground_truth['Participant No.'] == f"test_{i}"]
            
            if true_rows.empty:
                true_diagnosis = "Unknown"
                ori_truth = "Unknown"
            else:
                true_diagnosis = true_rows['Processed Diagnosis'].values[0]
                # 兼容不同列名
                if 'Diagnoses (related to pain)' in true_rows:
                    ori_truth = true_rows['Diagnoses (related to pain)'].values[0]
                else:
                    ori_truth = true_diagnosis

            # --- E. 生成混合报告 ---
            generated_report = generate_diagnosis_report_hybrid(
                augmented_features_path, 
                query_text, 
                str(retrieved_docs_content), 
                kare_context_text, 
                i, 
                top_n=1, 
                match_n=5, 
                model=my_model
            )

            # --- F. 提取诊断结论 ---
            generated_diagnosis = re.findall(r'\*\*Diagnosis\*\*:\s*(.*?)(?:\.|\n|$)', generated_report)
            pred = generated_diagnosis[0].strip() if generated_diagnosis else "Extraction Failed"

            results.append([participant_no, pred, true_diagnosis, ori_truth, generated_report])
            
            # --- G. 实时保存 ---
            # 每次循环都保存，防止程序中断丢失数据
            pd.DataFrame(results, columns=['Participant No.', 'Generated Diagnosis', 'True Diagnosis', 'Ori Truth', 'Report']).to_csv(output_filename, index=False)

        except Exception as e:
            print(f"\nError processing {participant_no}: {e}")
            # import traceback
            # traceback.print_exc()

    print(f"Done! Results saved to {output_filename}")

if __name__ == "__main__":
    main()