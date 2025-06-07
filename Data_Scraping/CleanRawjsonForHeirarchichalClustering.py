
import json
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1) Filenames (modify if needed)
# ─────────────────────────────────────────────────────────────────────────────
RAW_JSON_FILENAME     = "Electronics.json"    # <-- Your raw scraped JSON
CLEANED_JSON_FILENAME = "cleaned.json"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Preprocessing function
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    - Lowercase
    - Replace any non-alphanumeric (except whitespace) with a space
    - Collapse multiple spaces
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', " ", text)
    text = re.sub(r'\s+', " ", text).strip()
    return text

# ─────────────────────────────────────────────────────────────────────────────
# 3) Clean & Append Logic
# ─────────────────────────────────────────────────────────────────────────────
def clean_and_append(raw_path: Path, cleaned_path: Path):
  
    # 3A) Load or initialize existing list
    if cleaned_path.exists():
        print(f"Appending to existing {cleaned_path}")
        with open(cleaned_path, "r", encoding="utf-8") as f_in:
            existing = json.load(f_in)
    else:
        print(f"No existing {cleaned_path}, creating new.")
        existing = []

    # 3B) Load raw entries
    print(f"Loading raw data from {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f_in:
        raw_entries = json.load(f_in)

    new_list = []
    for entry in raw_entries:
        asin = entry.get("asin", "").strip()
        title = (entry.get("title") or "").strip()            # keep raw title
        brand = (entry.get("brand") or "").strip()
        breadcrumbs_raw = (entry.get("breadCrumbs") or "").strip()
        description = (entry.get("description") or "").strip()

        # Parse breadcrumbs into a list
        if breadcrumbs_raw:
            breadcrumbs_list = [seg.strip() for seg in re.split(r"[>/]", breadcrumbs_raw) if seg.strip()]
        else:
            breadcrumbs_list = []

        # Combine attributes "key value"
        attr_texts = []
        for attr in entry.get("attributes", []):
            key = (attr.get("key") or "").strip()
            val = (attr.get("value") or "").strip()
            if val:
                attr_texts.append(f"{key} {val}")

        # Combine ALL pieces into one string for cleaning
        combined = " ".join([
            title,
            brand,
            " ".join(breadcrumbs_list),
            description,
            " ".join(attr_texts)
        ]).strip()

        clean_text = preprocess_text(combined)

        cleaned_obj = {
            "asin": asin,
            "title": title,
            "clean_text": clean_text,
            "breadcrumbs": breadcrumbs_list
        }
        new_list.append(cleaned_obj)

    existing.extend(new_list)
    print(f"Appending {len(new_list)} items to cleaned.json (now {len(existing)} total).")

    with open(cleaned_path, "w", encoding="utf-8") as f_out:
        json.dump(existing, f_out, ensure_ascii=False, indent=2)
    print(f"Written updated cleaned data to {cleaned_path}")

    return existing

# ─────────────────────────────────────────────────────────────────────────────
# 4) Run if script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_path = Path(RAW_JSON_FILENAME)
    cleaned_path = Path(CLEANED_JSON_FILENAME)
    clean_and_append(raw_path, cleaned_path)
