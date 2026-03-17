import os

# --- Paths ---
DATA_DIR = "./dataset/ddxplus_data"
OUTPUT_DIR = "./dataset/kg_build"

# Raw Inputs
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "release_train_patients/release_train_patients")
TEST_DATA_PATH = os.path.join(DATA_DIR, "release_test_patients/release_test_patients")
CONDITIONS_PATH = os.path.join(DATA_DIR, "release_conditions.json")

# Intermediate & Final Outputs
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SYMPTOM_STATS_PATH = os.path.join(OUTPUT_DIR, "symptom_stats.json")
GRAPH_PICKLE_PATH = os.path.join(OUTPUT_DIR, "base_graph.gpickle")
AUGMENTED_GRAPH_PATH = os.path.join(OUTPUT_DIR, "augmented_graph.gpickle")

# Final Artifacts for MedRAG
FINAL_KG_XLSX = "./dataset/knowledge graph of DDXPlus.xlsx"
FINAL_GT_CSV = "./dataset/AI Data Set with Categories.csv"

# --- OpenAI Configuration ---
# Replace with your actual key or import from authentication
API_KEY = "sk-2T4AwFVk3OUPJhQaVmQfk7VvkfjxgVqVYRolJsiesL4zJcgs"
BASE_URL = "https://poloai.top/v1" 

# --- Tier 2 Taxonomy Mapping (Exact match with symptom_stats.json) ---
# Mapping: Pathology (Tier 3) -> (Tier 1 System, Tier 2 Subcategory)
DISEASE_TAXONOMY = {
    # Respiratory System
    "Acute COPD exacerbation / infection": ("Respiratory System", "COPD"),
    "Acute laryngitis": ("Respiratory System", "Upper Respiratory"),
    "Acute otitis media": ("Respiratory System", "Upper Respiratory"), # »ò ENT
    "Acute rhinosinusitis": ("Respiratory System", "Sinusitis"),
    "Allergic sinusitis": ("Respiratory System", "Allergy"),
    "Bronchiectasis": ("Respiratory System", "Bronchial Disease"),
    "Bronchiolitis": ("Respiratory System", "Infection"),
    "Bronchitis": ("Respiratory System", "Infection"),
    "Bronchospasm / acute asthma exacerbation": ("Respiratory System", "Asthma"),
    "Chronic rhinosinusitis": ("Respiratory System", "Sinusitis"),
    "Croup": ("Respiratory System", "Upper Respiratory"),
    "Epiglottitis": ("Respiratory System", "Upper Respiratory"),
    "Influenza": ("Respiratory System", "Viral Infection"),
    "Larygospasm": ("Respiratory System", "Other"),
    "Pneumonia": ("Respiratory System", "Infection"),
    "Pulmonary neoplasm": ("Respiratory System", "Neoplasm"),
    "Spontaneous pneumothorax": ("Respiratory System", "Pleural Disease"),
    "Spontaneous rib fracture": ("Respiratory System", "Trauma"),
    "Tuberculosis": ("Respiratory System", "Infection"),
    "URTI": ("Respiratory System", "Upper Respiratory"),
    "Viral pharyngitis": ("Respiratory System", "Upper Respiratory"),
    "Whooping cough": ("Respiratory System", "Infection"),

    # Cardiovascular System
    "Acute pulmonary edema": ("Cardiovascular System", "Heart Failure"),
    "Atrial fibrillation": ("Cardiovascular System", "Arrhythmia"),
    "Localized edema": ("Cardiovascular System", "Edema"),
    "Myocarditis": ("Cardiovascular System", "Inflammation"),
    "PSVT": ("Cardiovascular System", "Arrhythmia"),
    "Pericarditis": ("Cardiovascular System", "Inflammation"),
    "Possible NSTEMI / STEMI": ("Cardiovascular System", "Ischemic Heart Disease"),
    "Pulmonary embolism": ("Cardiovascular System", "Vascular Disease"),
    "Stable angina": ("Cardiovascular System", "Ischemic Heart Disease"),
    "Unstable angina": ("Cardiovascular System", "Ischemic Heart Disease"),

    # Gastrointestinal System
    "Boerhaave": ("Gastrointestinal System", "Emergency"),
    "GERD": ("Gastrointestinal System", "Esophageal"),
    "Inguinal hernia": ("Gastrointestinal System", "Hernia"),
    "Pancreatic neoplasm": ("Gastrointestinal System", "Neoplasm"),
    "Scombroid food poisoning": ("Gastrointestinal System", "Poisoning"),

    # Neurological / Psychiatric
    "Acute dystonic reactions": ("Neurological System", "Movement Disorder"),
    "Cluster headache": ("Neurological System", "Headache"),
    "Guillain-Barre syndrome": ("Neurological System", "Neuropathy"),
    "Myasthenia gravis": ("Neurological System", "Neuromuscular"),
    "Panic attack": ("Neurological System", "Psychiatric"),

    # Immune / Hematologic / Systemic
    "Anaphylaxis": ("Immune System", "Allergy"),
    "Anemia": ("Hematologic System", "Anemia"),
    "Chagas": ("Systemic", "Infection"),
    "Ebola": ("Systemic", "Viral Infection"),
    "HIV (initial infection)": ("Immune System", "Infection"),
    "SLE": ("Immune System", "Autoimmune"),
    "Sarcoidosis": ("Multi-system", "Inflammation"),
    
    # Fallback
    "default": ("General System", "Unclassified")
}