
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# ─────────────────────────────────────────────────────────────────────────────
# 1) Filenames (current working directory)
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
    Each item in cleaned.json has keys:
      - "asin"
      - "title"
      - "clean_text"
      - "breadcrumbs" (list of category segments, e.g. ["Electronics","Cell Phones & Accessories","Cell Phones & Smartphones"])
    """
    print(f"Loading cleaned data from: {path}")
    with open(path, "r", encoding="utf-8") as f_in:
        return json.load(f_in)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Fit TF–IDF on entire clean_text corpus
# ─────────────────────────────────────────────────────────────────────────────
def fit_tfidf(cleaned_data, max_features=1000):
    """
    Returns:
      - tfidf_matrix (N × V)
      - terms (length V)
      - vectorizer (the fitted TF–IDF object)
    """
    corpus = [item["clean_text"] for item in cleaned_data]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_sp = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    tfidf_mat = tfidf_sp.toarray()
    print(f"TF–IDF fitted: N={len(corpus)}, V={len(terms)}")
    return tfidf_mat, terms, vectorizer

# ─────────────────────────────────────────────────────────────────────────────
# 4) Train or load Doc2Vec
# ─────────────────────────────────────────────────────────────────────────────
def train_doc2vec(cleaned_data, vector_size=100, window=5, min_count=2, epochs=40):
    """
    Train a Doc2Vec (PV-DM) on each product’s 'clean_text', tagging by str(i).
    Save to DOC2VEC_MODEL_FILENAME.
    """
    tagged_docs = []
    for idx, item in enumerate(cleaned_data):
        tokens = item["clean_text"].split()
        tagged_docs.append(TaggedDocument(tokens, [str(idx)]))

    print("Building Doc2Vec vocabulary...")
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        dm=1
    )
    model.build_vocab(tagged_docs)
    print(f"Training Doc2Vec for {epochs} epochs...")
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(DOC2VEC_MODEL_FILENAME)
    print(f"Saved Doc2Vec model to: {DOC2VEC_MODEL_FILENAME}")
    return model

def load_or_train_doc2vec(cleaned_data):
    path = Path(DOC2VEC_MODEL_FILENAME)
    if path.exists():
        print(f"Loading existing Doc2Vec from: {path}")
        return Doc2Vec.load(DOC2VEC_MODEL_FILENAME)
    else:
        return train_doc2vec(cleaned_data)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Extract each product’s Doc2Vec vector (N×D)
# ─────────────────────────────────────────────────────────────────────────────
def get_doc_vectors(model, num_docs):
    vectors = []
    for i in range(num_docs):
        vectors.append(model.dv[str(i)])
    return np.vstack(vectors)

# ─────────────────────────────────────────────────────────────────────────────
# 6) Build Ward linkage tree & plot truncated dendrogram with auto‐cut
# ─────────────────────────────────────────────────────────────────────────────
def build_and_plot_hierarchy(doc_vectors, cleaned_data, save_path: Path, truncate_level=5):
    """
    1) Compute Z = linkage(doc_vectors, method="ward").
    2) Plot a truncated dendrogram (showing top 'truncate_level' merges),
       using the first 20 chars of each product’s title as leaf labels.
    3) Determine a “cut threshold” by finding the largest gap in Z[:,2].
    4) Draw a red dashed line at that threshold.
    5) Return (Z, threshold).
    """
    print("Computing Ward linkage on document vectors...")
    Z = linkage(doc_vectors, method="ward")

    # 2A) Prepare leaf labels (first 20 chars of title)
    labels = []
    for item in cleaned_data:
        t = item["title"]
        if len(t) > 20:
            labels.append(t[:20] + "…")
        else:
            labels.append(t)

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

    # 2B) Automatically choose threshold based on largest gap
    distances = Z[:, 2]            # the distance values at each merge
    diffs = np.diff(distances)     # gaps between consecutive distances
    idx_max = np.argmax(diffs)     # index at which gap is largest
    threshold = (distances[idx_max] + distances[idx_max + 1]) / 2.0
    print(f"Automatically chosen threshold = {threshold:.4f} (gap index = {idx_max})")

    # 2C) Draw a horizontal line at threshold
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"cut @ {threshold:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    print(f"Dendrogram saved to: {save_path}")
    plt.show()

    return Z, threshold

# ─────────────────────────────────────────────────────────────────────────────
# 7) Derive a “category‐based” label from breadcrumbs for each cluster
# ─────────────────────────────────────────────────────────────────────────────
def get_cluster_category_label(idxs, cleaned_data):
    """
    1) For each index in idxs, extract cleaned_data[i]["breadcrumbs"].
    2) If a product has breadcrumbs = ["Electronics","Cell Phones & Accessories","Cell Phones & Smartphones"],
       we pick the LAST element: "Cell Phones & Smartphones".
    3) Count frequencies of these LAST elements across all idxs.  
    4) If at least one appears, pick the most common; otherwise return None.
    """
    last_categories = []
    for i in idxs:
        b = cleaned_data[i]["breadcrumbs"]
        if b:
            last_categories.append(b[-1])
    if not last_categories:
        return None
    most_common = Counter(last_categories).most_common(1)[0][0]
    return most_common

# ─────────────────────────────────────────────────────────────────────────────
# 8) Derive a TF–IDF fallback label if breadcrumbs fail
# ─────────────────────────────────────────────────────────────────────────────
def get_cluster_tfidf_label(idxs, tfidf_matrix, terms, top_k=1):
    submat = tfidf_matrix[idxs, :]
    sums = submat.sum(axis=0)
    top_indices = np.argsort(sums)[-top_k:][::-1]
    top_terms = [terms[i] for i in top_indices if sums[i] > 0]
    if not top_terms:
        return "misc"
    return " ".join(top_terms)

# ─────────────────────────────────────────────────────────────────────────────
# 9) Build the multi‐child JSON by cutting at threshold
# ─────────────────────────────────────────────────────────────────────────────
def build_multi_child_hierarchy(cleaned_data, doc_vectors, tfidf_matrix, terms, Z, threshold):
    """
    1) Form clusters with fcluster(Z, t=threshold, criterion="distance").  
    2) For each resulting cluster ID (cid):
         - idxs = list of product‐indices in that cluster
         - Try to label = get_cluster_category_label(idxs, cleaned_data)
         - If None, fallback to get_cluster_tfidf_label(idxs, tfidf_matrix, terms)
         - Compute centroid = mean(doc_vectors[idxs, :])
         - members = [cleaned_data[i]["asin"] for i in idxs]
       Construct:
         {
           "cluster_id": <cid>,
           "name": <label>,
           "centroid": [...100 floats...],
           "members": [<ASIN1>, <ASIN2>, …]
         }
    3) Return { "root": { "children": [cluster_obj, …] } }.
    """
    num_docs = len(cleaned_data)
    cluster_labels = fcluster(Z, t=threshold, criterion="distance")
    unique_cids = np.unique(cluster_labels)
    print(f"Number of top‐level clusters formed: {len(unique_cids)}")

    dv = np.asarray(doc_vectors)
    clusters = []
    for cid in unique_cids:
        idxs = np.where(cluster_labels == cid)[0].tolist()

        # 9A) Try category‐based label from breadcrumbs
        label = get_cluster_category_label(idxs, cleaned_data)
        if label is None:
            # 9B) Fallback to TF–IDF‐based term
            label = get_cluster_tfidf_label(idxs, tfidf_matrix, terms, top_k=1)

        # 9C) Centroid = average of doc_vectors[idxs]
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

    # 10B) Fit TF–IDF on 'clean_text'
    tfidf_matrix, terms, vectorizer = fit_tfidf(cleaned_data, max_features=1000)

    # 10C) Train or load Doc2Vec
    model = load_or_train_doc2vec(cleaned_data)

    # 10D) Extract Doc2Vec embeddings (N×D)
    print("Extracting Doc2Vec vectors...")
    doc_vectors = get_doc_vectors(model, N)

    # 10E) Build Ward linkage tree & plot dendrogram + auto cut‐threshold
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
