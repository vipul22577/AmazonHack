import json
import re
import numpy as np
from pathlib import Path

from gensim.models.doc2vec import Doc2Vec
from scipy.spatial.distance import cosine

# ─────────────────────────────────────────────────────────────────────────────
# 1) Filenames (in the current working directory)
# ─────────────────────────────────────────────────────────────────────────────
CLEANED_JSON_FILENAME   = "cleaned.json"
DOC2VEC_MODEL_FILENAME  = "doc2vec.model"
TREE_JSON_FILENAME      = "product_tree.json"
TEST_RAW_FILENAME       = "Electronics.json"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load doc2vec model & multi-child tree JSON
# ─────────────────────────────────────────────────────────────────────────────
print("Loading Doc2Vec model from:", DOC2VEC_MODEL_FILENAME)
doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL_FILENAME)

print("Loading multi-child tree from:", TREE_JSON_FILENAME)
with open(TREE_JSON_FILENAME, "r", encoding="utf-8") as f_tree:
    multi_tree = json.load(f_tree)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Build mapping: asin → (cluster_id, cluster_name, cluster_centroid)
# ─────────────────────────────────────────────────────────────────────────────
cluster_map = {}   # asin → (cluster_id, cluster_name)
centroids = {}     # cluster_id → np.array(centroid)

for child in multi_tree["root"]["children"]:
    cid = int(child["cluster_id"])
    cname = child["name"]
    cent = np.array(child["centroid"])
    centroids[cid] = cent
    for asin in child["members"]:
        cluster_map[asin] = (cid, cname)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Preprocess incoming text (same logic as Cell 1)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', " ", text)
    text = re.sub(r'\s+', " ", text).strip()
    return text

def build_cleaned_text(entry: dict) -> str:
    """
    Given raw entry dict (with keys: asin, title, brand, breadCrumbs, description, attributes),
    build the cleaned string exactly as Cell 1 did.
    """
    title = (entry.get("title") or "").strip()
    brand = (entry.get("brand") or "").strip()
    breadcrumbs_raw = (entry.get("breadCrumbs") or "").strip()
    description = (entry.get("description") or "").strip()

    if breadcrumbs_raw:
        breadcrumbs_list = [seg.strip() for seg in re.split(r"[>/]", breadcrumbs_raw) if seg.strip()]
    else:
        breadcrumbs_list = []

    attr_texts = []
    for attr in entry.get("attributes", []):
        key = (attr.get("key") or "").strip()
        val = (attr.get("value") or "").strip()
        if val:
            attr_texts.append(f"{key} {val}")

    combined = " ".join([
        title,
        brand,
        " ".join(breadcrumbs_list),
        description,
        " ".join(attr_texts)
    ]).strip()

    return preprocess_text(combined)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Classification function: assign one cleaned product text to a cluster_id
# ─────────────────────────────────────────────────────────────────────────────
def classify_to_cluster(clean_text: str):
    """
    - Infer Doc2Vec vector
    - Compute cosine distance to each cluster centroid
    - Return the cluster_id and cluster_name with minimum distance
    """
    tokens = clean_text.split()
    vec = doc2vec_model.infer_vector(tokens)

    best_cid = None
    best_dist = float("inf")
    for cid, cent in centroids.items():
        dist = cosine(vec, cent)
        if dist < best_dist:
            best_dist = dist
            best_cid = cid
    # Get cluster_name from `multi_tree["root"]["children"]` or cluster_map
    cname = None
    for child in multi_tree["root"]["children"]:
        if int(child["cluster_id"]) == best_cid:
            cname = child["name"]
            break
    return best_cid, cname

# ─────────────────────────────────────────────────────────────────────────────
# 6) Main: Load test_raw.json, classify each, compute accuracy
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_raw_path = Path(TEST_RAW_FILENAME)
    print(f"Loading test entries from: {test_raw_path}")
    with open(test_raw_path, "r", encoding="utf-8") as f_test:
        test_entries = json.load(f_test)

    total = len(test_entries)
    correct = 0

    print("\nClassifying test entries. Comparing predicted vs. true clusters:\n")
    for idx, entry in enumerate(test_entries):
        test_asin = entry.get("asin", "").strip()
        clean_text = build_cleaned_text(entry)
        pred_cid, pred_name = classify_to_cluster(clean_text)

        # Determine “true” cluster by looking up test_asin in cluster_map
        if test_asin in cluster_map:
            true_cid, true_name = cluster_map[test_asin]
        else:
            true_cid, true_name = (None, "UNKNOWN")

        if pred_cid == true_cid and true_cid is not None:
            correct += 1
            status = "✔️"
        else:
            status = "❌"

        print(
            f"[{idx:03d}] ASIN={test_asin} : Predicted → (cid={pred_cid}, '{pred_name}')  |  "
            f"True → (cid={true_cid}, '{true_name}')  {status}"
        )

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n=== Classification Accuracy: {correct} / {total} = {accuracy:.2%} ===")
