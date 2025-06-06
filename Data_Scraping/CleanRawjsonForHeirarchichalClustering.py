import json
from pathlib import Path

# Load the raw JSON data
input_path = Path("dataset_Amazon-crawler_2025-06-06_13-24-03-537.json")
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned = []
for entry in raw_data:
    # Extract fields
    asin = entry.get("asin", "")
    title = entry.get("title", "") or ""
    brand = entry.get("brand", "") or ""
    bread_crumbs = entry.get("breadCrumbs", "") or ""
    description = entry.get("description", "") or ""

    # Collect attribute values
    attributes = entry.get("attributes", [])
    attr_texts = []
    for attr in attributes:
        key = attr.get("key", "")
        value = attr.get("value", "")
        if value:
            attr_texts.append(f"{key}: {value}")

    # Combine texts into a single field
    combined_text = " ".join([title, brand, bread_crumbs, description, " ".join(attr_texts)])

    # Append cleaned entry
    cleaned.append({
        "asin": asin,
        "clean_text": combined_text.strip()
    })

# Write cleaned JSON to file
output_path = Path("cleaned.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)
