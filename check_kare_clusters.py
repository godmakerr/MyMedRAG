import pickle

with open("data/kare_community_index.pkl", "rb") as f:
    clusters = pickle.load(f)

print(f"Total Clusters: {len(clusters)}\n")

for c in clusters:
    desc = c['description']
    summary = desc.split('.')[0] 
    print(f"[ID {c['id']}] {summary}")