import json
import os
from pathlib import Path

RAW_PATH  = Path("dataset_Amazon-crawler_2025-06-06_13-24-03-537.json")
CLEANED_PATH = Path("cleaned.json")

def write_clean_entry(clean_obj, out_path: Path):
    """
    Appends a single JSON object to out_path, maintaining valid JSON array format.
    - If out_path does not exist, create it as “[ {...} ]”.
    - If out_path already exists, insert “,\n <new_obj> \n]” right before the existing closing bracket.
    """
    entry_str = json.dumps(clean_obj, ensure_ascii=False)
    if not out_path.exists():
        # First time: create new file, write “[ \n <entry> \n ]”
        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.write("[\n")
            f_out.write(entry_str)
            f_out.write("\n]")
    else:
        # Append: open in r+ mode, seek back to find the final ‘]’, then insert “, <entry> ]”
        with open(out_path, "r+", encoding="utf-8") as f_out:
            # Move pointer to the very end
            f_out.seek(0, os.SEEK_END)
            pos = f_out.tell()

            # Step backwards until we find the first ‘]’ character (ignoring whitespace).
            # Once found, we’ll overwrite starting at that position.
            while pos > 0:
                pos -= 1
                f_out.seek(pos)
                if f_out.read(1) == "]":
                    break

            # Now pos is the index of the final ‘]’.
            f_out.seek(pos)
            f_out.truncate()                # erase from pos onward
            f_out.write(",\n")              # comma + newline to separate array items
            f_out.write(entry_str)          # the new object
            f_out.write("\n]")              # close the JSON array

def clean_and_append_all():
    # Open the raw file once, load it as a Python list. (If it's truly enormous,
    # you could replace this with a streaming parser, but in most cases this
    # is acceptable. The “on‐the‐go” part is in how we write.)
    with open(RAW_PATH, "r", encoding="utf-8") as f_raw:
        raw_entries = json.load(f_raw)

    for entry in raw_entries:
        # 1) Extract the ASIN
        asin = entry.get("asin", "").strip()

        # 2) Fields to keep for “clean_text”
        title       = (entry.get("title")       or "").strip()
        brand       = (entry.get("brand")       or "").strip()
        breadcrumbs = (entry.get("breadCrumbs") or "").strip()
        description = (entry.get("description") or "").strip()

        # 3) Pull out every non‐empty “attributes” value
        attr_texts = []
        for attr in entry.get("attributes", []):
            key   = attr.get("key", "").strip()
            value = attr.get("value", "").strip()
            if value:
                # e.g. "Product Dimensions: 30 x 25 x 2 cm; 300 g"
                attr_texts.append(f"{key}: {value}")

        # 4) Build one large string
        combined_text = " ".join([
            title,
            brand,
            breadcrumbs,
            description,
            " ".join(attr_texts)
        ]).strip()

        # 5) Create the “clean object”
        clean_obj = {
            "asin": asin,
            "clean_text": combined_text
        }

        # 6) Append it to cleaned.json (creating the file first if needed)
        write_clean_entry(clean_obj, CLEANED_PATH)

if __name__ == "__main__":
    clean_and_append_all()
