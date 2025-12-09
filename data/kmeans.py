import numpy as np
from sklearn.cluster import KMeans
import json
dataset = "Games"
for emb_path in [f"{dataset}/{dataset}.emb-llama-td.npy", f"{dataset}/{dataset}.emb-ViT-L-14.npy"]:
    embeddings = np.load(emb_path)
    print("Embedding shape:", embeddings.shape)


    n_clusters = 512
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)


    item2cluster = {}
    for idx, label in enumerate(labels):
        item2cluster[f"{idx}"] = [f"<a_{label}>"]


    if "llama" in emb_path:
        out_path = f"{dataset}/{dataset}.index_lemb_kmeans{n_clusters}.json"
    else:
        out_path = f"{dataset}/{dataset}.index_vitemb_kmeans{n_clusters}.json"
    with open(out_path, "w") as f:
        json.dump(item2cluster, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_path}")
