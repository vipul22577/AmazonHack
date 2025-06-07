# Cell 2: build_multi_child_tree_hierarchical_with_retrain_check.py

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# ─────────────────────────────────────────────────────────────────────────────
# 1) Filenames (in the current working directory)
# ─────────────────────────────────────────────────────────────────────────────
CLEANED_JSON_FILENAME  = "cleaned.json"
DOC2VEC_MODEL_FILENAME = "doc2vec.model"
TREE_JSON_FILENAME     = "product_tree.json"
DENDROGRAM_PNG         = "dendrogram.png"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load cleaned.json
# ─────────────────────────────────────────────────────────────────────────────
def load_cleaned_data(path: Path):
    """
    Each item has:
      - "asin"
      - "title"
      - "clean_text"
      - "breadcrumbs" (list of category segments)
    """
    print(f"Loading cleaned data from: {path}")
    with open(path, "r", encoding="utf-8") as f_in:
        return json.load(f_in)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Fit TF–IDF on clean_text corpus
# ─────────────────────────────────────────────────────────────────────────────
def fit_tfidf(cleaned_data, max_features=1000):
    """
    Returns (tfidf_matrix [N×V], terms [length V], vectorizer).
    """
    corpus = [item["clean_text"] for item in cleaned_data]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_sp = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    tfidf_mat = tfidf_sp.toarray()
    print(f"TF–IDF fitted: N={len(corpus)}, V={len(terms)}")
    return tfidf_mat, terms, vectorizer

# ─────────────────────────────────────────────────────────────────────────────
# 4) Train (or retrain) Doc2Vec if needed
# ─────────────────────────────────────────────────────────────────────────────
def train_doc2vec(cleaned_data, vector_size=100, window=5, min_count=2, epochs=40):
    """
    Always trains a fresh Doc2Vec model on the full cleaned_data, tagging by str(i).
    Saves to DOC2VEC_MODEL_FILENAME.
    """
    print("Training a new Doc2Vec model on all cleaned entries...")
    tagged_docs = []
    for idx, item in enumerate(cleaned_data):
        tokens = item["clean_text"].split()
        tagged_docs.append(TaggedDocument(tokens, [str(idx)]))

    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        dm=1
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(DOC2VEC_MODEL_FILENAME)
    print(f"Doc2Vec model saved to: {DOC2VEC_MODEL_FILENAME} (trained on N={len(cleaned_data)})")
    return model

def load_or_retrain_doc2vec(cleaned_data):
    """
    - If DOC2VEC_MODEL_FILENAME does not exist, train a new model.
    - If it does exist, load and check if model.dv.count == len(cleaned_data).
      • If equal, return the loaded model.
      • If not, retrain on the full cleaned_data and overwrite.
    """
    model_path = Path(DOC2VEC_MODEL_FILENAME)
    N = len(cleaned_data)
    if not model_path.exists():
        return train_doc2vec(cleaned_data)

    print(f"Loading existing Doc2Vec model from: {model_path}")
    model = Doc2Vec.load(DOC2VEC_MODEL_FILENAME)
    if len(model.dv) != N:
        print(f"Doc2Vec model has {len(model.dv)} docs, but cleaned.json has {N}. Retraining.")
        return train_doc2vec(cleaned_data)
    else:
        print("Doc2Vec model matches cleaned data length. No retraining needed.")
        return model

# ─────────────────────────────────────────────────────────────────────────────
# 5) Extract Doc2Vec embeddings (N × vector_size)
# ─────────────────────────────────────────────────────────────────────────────
def get_doc_vectors(model, num_docs):
    """
    Returns array of shape (num_docs, vector_size).
    Throws KeyError if the model does not have tag str(i) for some i.
    """
    vectors = []
    for i in range(num_docs):
        vectors.append(model.dv[str(i)])
    return np.vstack(vectors)

# ─────────────────────────────────────────────────────────────────────────────
# 6) Build Ward linkage & plot truncated dendrogram with auto‐cut
# ─────────────────────────────────────────────────────────────────────────────
def build_and_plot_hierarchy(doc_vectors, cleaned_data, save_path: Path, truncate_level=5):
    """
    1) Compute Z = linkage(doc_vectors, method="ward").
    2) Plot a truncated dendrogram (top 'truncate_level' merges),
       labeling leaves by the first 20 characters of each product title.
    3) Determine auto‐cut threshold by largest gap in Z[:,2].
    4) Draw a horizontal line at threshold in red dashed.
    5) Return (Z, threshold).
    """
    print("Computing Ward linkage on document vectors...")
    Z = linkage(doc_vectors, method="ward")

    # 2A) Create leaf labels: first 20 chars of title + "…" if longer
    labels = []
    for item in cleaned_data:
        t = item["title"]
        labels.append((t[:20] + "…") if len(t) > 20 else t)

    print(f"Plotting truncated dendrogram (top {truncate_level} merges)...")
    plt.figure(figsize=(14, 6))
    dendrogram(
        Z,
        truncate_mode="level",
        p=truncate_level,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=6,
        show_contracted=True
    )
    plt.title("Hierarchical Clustering Dendrogram (Truncated)")
    plt.xlabel("Product Title (first 20 chars)")
    plt.ylabel("Ward Distance")

    # 2B) Automatically pick threshold at largest gap
    distances = Z[:, 2]            # distances for each merge
    diffs = np.diff(distances)     # gaps between successive distances
    idx_max = np.argmax(diffs)
    threshold = (distances[idx_max] + distances[idx_max + 1]) / 2.0
    print(f"Auto‐chosen threshold = {threshold:.4f} (largest gap at index {idx_max})")

    # 2C) Draw horizontal line
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"cut @ {threshold:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    print(f"Dendrogram saved to: {save_path}")
    plt.show()

    return Z, threshold

# ─────────────────────────────────────────────────────────────────────────────
# 7) Label clusters by most‐specific breadcrumb (last element)
# ─────────────────────────────────────────────────────────────────────────────
def get_cluster_category_label(idxs, cleaned_data):
    """
    - idxs: list of product indices belonging to a cluster.
    - For each i in idxs, fetch cleaned_data[i]["breadcrumbs"].
    - last_breadcrumb = b[-1] if b is nonempty.
    - Count frequencies of those last_breadcrumbs; return the most common.
    - If no breadcrumbs or all empty, return None.
    """
    last_cats = []
    for i in idxs:
        b = cleaned_data[i]["breadcrumbs"]
        if b:
            last_cats.append(b[-1])
    if not last_cats:
        return None
    return Counter(last_cats).most_common(1)[0][0]

# ─────────────────────────────────────────────────────────────────────────────
# 8) Fallback: label by top TF–IDF term if breadcrumbs are missing
# ─────────────────────────────────────────────────────────────────────────────
def get_cluster_tfidf_label(idxs, tfidf_matrix, terms, top_k=1):
    """
    - Sum TF–IDF rows for idxs: sums length‐V vector.
    - Pick top_k terms from sums. Return "term1 term2" if top_k=2.
    - If no positive sums, return "misc".
    """
    submat = tfidf_matrix[idxs, :]
    sums = submat.sum(axis=0)
    top_indices = np.argsort(sums)[-top_k:][::-1]
    top_terms = [terms[i] for i in top_indices if sums[i] > 0]
    if not top_terms:
        return "misc"
    return " ".join(top_terms)

# ─────────────────────────────────────────────────────────────────────────────
# 9) Build multi‐child tree by cutting dendrogram at threshold
# ─────────────────────────────────────────────────────────────────────────────
def build_multi_child_hierarchy(cleaned_data, doc_vectors, tfidf_matrix, terms, Z, threshold):
    """
    1) cluster_labels = fcluster(Z, t=threshold, criterion="distance")
    2) unique_cids = unique(cluster_labels)
    3) For each cid:
         - idxs = [i for i in range(N) if cluster_labels[i] == cid]
         - Try label = get_cluster_category_label(idxs, cleaned_data)
         - If None, label = get_cluster_tfidf_label(idxs, tfidf_matrix, terms)
         - centroid = mean(doc_vectors[idxs, :]) as list
         - members = [cleaned_data[i]["asin"] for i in idxs]
       Create cluster_obj with keys:
         { "cluster_id": cid, "name": label, "centroid": centroid, "members": members }
    4) Return { "root": { "children": [ cluster_obj, ... ] } }.
    """
    N = len(cleaned_data)
    print("Forming clusters by fcluster at threshold:", threshold)
    cluster_labels = fcluster(Z, t=threshold, criterion="distance")
    unique_cids = np.unique(cluster_labels)
    print(f"Number of top‐level clusters: {len(unique_cids)}")

    dv = np.array(doc_vectors)
    clusters = []
    for cid in unique_cids:
        idxs = np.where(cluster_labels == cid)[0].tolist()
        # 9A) Category label by breadcrumbs
        label = get_cluster_category_label(idxs, cleaned_data)
        if label is None:
            # 9B) TF–IDF fallback
            label = get_cluster_tfidf_label(idxs, tfidf_matrix, terms, top_k=1)
        # 9C) Compute centroid
        centroid = dv[idxs, :].mean(axis=0).tolist()
        # 9D) Member ASINs
        members = [cleaned_data[i]["asin"] for i in idxs]
        clusters.append({
            "cluster_id": int(cid),
            "name": label,
            "centroid": centroid,
            "members": members
        })

    return { "root": { "children": clusters } }

# ─────────────────────────────────────────────────────────────────────────────
# 10) Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 10A) Load cleaned data
    cleaned_path = Path(CLEANED_JSON_FILENAME)
    cleaned_data = load_cleaned_data(cleaned_path)
    N = len(cleaned_data)
    print(f"Total products: {N}")

    # 10B) Fit TF–IDF
    tfidf_matrix, terms, vectorizer = fit_tfidf(cleaned_data, max_features=1000)

    # 10C) Load or retrain Doc2Vec
    model = load_or_retrain_doc2vec(cleaned_data)

    # 10D) Extract Doc2Vec embeddings
    print("Extracting Doc2Vec vectors...")
    doc_vectors = get_doc_vectors(model, N)

    # 10E) Build Ward tree & plot dendrogram with auto cut
    Z, threshold = build_and_plot_hierarchy(
        doc_vectors,
        cleaned_data,
        save_path=Path(DENDROGRAM_PNG),
        truncate_level=5
    )

    # 10F) Build multi‐child hierarchy by cutting at threshold
    print("Building multi‐child hierarchy (cut at threshold).")
    tree_dict = build_multi_child_hierarchy(
        cleaned_data,
        doc_vectors,
        tfidf_matrix,
        terms,
        Z,
        threshold
    )

    # 10G) Save final tree JSON
    out_path = Path(TREE_JSON_FILENAME)
    print(f"Saving multi‐child tree JSON to: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(tree_dict, f_out, ensure_ascii=False, indent=2)
    print(f"Multi‐child hierarchy saved → {TREE_JSON_FILENAME}")

    print("✅ build_multi_child_tree_hierarchical_with_breadcrumbs.py completed.")
    print(f"  • Doc2Vec model → {DOC2VEC_MODEL_FILENAME}")
    print(f"  • Tree JSON     → {TREE_JSON_FILENAME}")
    print(f"  • Dendrogram PNG → {DENDROGRAM_PNG}")
