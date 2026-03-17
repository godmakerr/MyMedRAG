import os
import re
import json
import pandas as pd
import numpy as np
import faiss  # <--- 直接在这里导入 faiss
from tqdm import tqdm
from main_MedRAG import (
    get_query_embedding, 
    # Faiss,  <--- 删掉这个引用，我们要自己写
    extract_diagnosis, 
    documents, 
    document_embeddings, 
    generate_diagnosis_report, 
    save_results_to_csv, 
    get_additional_info_from_level_2, 
    KG_preprocess, 
    get_embeddings
)
from authentication import (
    ob_path, 
    test_folder_path, 
    ground_truth_file_path, 
    augmented_features_path
)

# --- 本地定义的无敌 Faiss 函数 ---
def LocalFaiss(doc_vecs, query_vec, k=1):
    # 确保 contiguous (连续内存)，这是 Faiss 的隐形要求
    doc_vecs = np.ascontiguousarray(doc_vecs)
    query_vec = np.ascontiguousarray(query_vec)
    
    # 建立索引
    index = faiss.IndexFlatL2(doc_vecs.shape[1])
    index.add(doc_vecs)
    D, I = index.search(query_vec, k)
    return I
# -------------------------------

# 疾病列表 (按需保留)
disease_list = [
    "Head pain", "Migraine", "Trigeminal neuralgia", "Cervical spondylosis", "Chronic neck pain", "Neck pain",
    "Chest pain", "Abdominal pain", "Limb pain", "Shoulder pain", "Hip pain", "Knee pain", "Buttock pain",
    "Calf pain", "Low back pain", "Chronic low back pain", "Mechanical low back pain", "Upper back pain",
    "Degenerative disc disease", "Lumbar spondylosis", "Lumbar canal stenosis", "Spinal stenosis", "Foraminal stenosis",
    "Lumbar_radicular_pain", "Radicular pain", "Sciatica", "Lumbosacral pain", "Generalized body pain", "Fibromyalgia",
    "Musculoskeletal pain", "Myofascial pain syndrome", "Neuropathic pain", "Post-herpetic neuralgia"
]

ground_truth = pd.read_csv(ground_truth_file_path, header=0)

results = []
if os.path.exists(test_folder_path):
    file_paths = os.listdir(test_folder_path)
else:
    file_paths = []
    print(f"Warning: Test folder not found at {test_folder_path}")

topk = 1
top_n = 1
match_n = 5
samplerange = range(1000, 1051) 
my_model = "gpt-4o" 

# --- 预处理知识库向量 (只做一次) ---
print("正在预处理知识库向量...")
try:
    # 1. 转 Numpy
    doc_vecs_cache = np.array(document_embeddings)
    # 2. 转 Float32
    doc_vecs_cache = doc_vecs_cache.astype('float32')
    # 3. 打印检查
    print(f"DEBUG: 知识库向量形状: {doc_vecs_cache.shape}, 类型: {doc_vecs_cache.dtype}")
except Exception as e:
    print(f"❌ 知识库向量转换失败: {e}")
    # 如果这里失败了，说明 document_embeddings 数据结构完全坏了（比如是空的，或者参差不齐）
    exit()

for i in tqdm(samplerange):
    print(f"\nProcessing patient {i}...")
    
    file_path = os.path.join(test_folder_path, f"participant_{i}.json")
    if not os.path.exists(file_path):
        print(f'{i} is not found')
        continue

    with open(file_path, 'r') as file:
        new_patient_case = json.load(file)

    participant_no = new_patient_case['Participant No.']
    query = json.dumps(new_patient_case)

    success = False
    while not success:
        try:
            # 1. 获取查询向量
            query_embedding = get_query_embedding(query)

            # 2. 转换查询向量
            query_vec = np.array(query_embedding).astype('float32')
            if len(query_vec.shape) == 1:
                query_vec = query_vec.reshape(1, -1)
            
            # DEBUG 打印 (如果报错，这行能救命)
            # print(f"Query shape: {query_vec.shape}, Doc shape: {doc_vecs_cache.shape}")

            # 3. 调用本地 Faiss (不再用 main_MedRAG 里的了)
            indices = LocalFaiss(doc_vecs_cache, query_vec, k=topk)
            
            retrieved_documents = [documents[idx] for idx in indices[0]]
            final_retrieved_info = []
            
            for retrieved_document in retrieved_documents:
                with open(retrieved_document, 'r') as file:
                    patient_case = json.load(file)
                    patient_case_json = json.dumps(patient_case)
                    patient_case_dict = json.loads(patient_case_json)
                    
                    filtered_patient_case_dict = {
                        key: patient_case_dict[key] for key in [
                            "Processed Diagnosis",
                            "Pain Presentation and Description Areas of pain as per physiotherapy input",
                            "Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles",
                            "Pain/General Physiotherapist Treatments (Treatments\nSession No.: General Overview\n- Specific interventions/treatments)",
                            "Pain Psychologist Treatments (Treatments)",
                            "Pain Medicine Treatments (Treatments)",
                        ] if key in patient_case_dict
                    }
                    final_retrieved_info.append(filtered_patient_case_dict)

            # 4. 匹配真实诊断
            true_diagnosis_row = ground_truth.loc[ground_truth['Participant No.'] == participant_no]
            if true_diagnosis_row.empty:
                true_diagnosis_row = ground_truth.loc[ground_truth['Participant No.'] == str(participant_no)]
            if true_diagnosis_row.empty:
                 true_diagnosis_row = ground_truth.loc[ground_truth['Participant No.'] == f"test_{participant_no}"]

            if true_diagnosis_row.empty:
                print(f"True diagnosis for patient_{participant_no} not found")
                break

            true_diagnosis = true_diagnosis_row['Processed Diagnosis'].values[0]
            if 'Diagnoses (related to pain)' in true_diagnosis_row:
                ori_truth = true_diagnosis_row['Diagnoses (related to pain)'].values[0]
            else:
                ori_truth = true_diagnosis

            # 5. 生成报告
            generated_report_ori = generate_diagnosis_report(
                augmented_features_path,
                query, 
                final_retrieved_info, 
                i, 
                top_n=top_n, 
                match_n=match_n,
                model=my_model 
            )
            # print(generated_report_ori) # <--- 改回这个，就能看到详细的诊断结果了！
            print("Report Generated Successfully.") # 简化日志

            generated_diagnosis = re.findall(r'\*\*Diagnosis\*\*:\s*(.*?)(?:\.|\n|$)', generated_report_ori)
            
            if not generated_diagnosis:
                results.append([participant_no, '', true_diagnosis, ori_truth, generated_report_ori])
                break
            else:
                print("Success!!!")

            results.append([participant_no, generated_diagnosis[0], true_diagnosis, ori_truth, generated_report_ori])
            success = True
            print('________________________________________________________________')
            
        except Exception as e:
            print(f"Error processing patient_{participant_no}: {e}")
            import traceback
            traceback.print_exc()
            break

output_file = f"./test_results_topk{topk}_topn{top_n}_matchn{match_n}_{samplerange.start}_{samplerange.stop}_MedRAG.csv"#  f"./test_results_MedRAG.csv"
df = pd.DataFrame(results, columns=['Participant No.', 'Generated Diagnosis', 'True Diagnosis', 'Ori Truth', 'Generated report'])
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")