import json
import csv
import os

# ===================== CONFIGURATION =====================
SURVEY = "pvq" # big5, pvq, via_part1, via_part2
# =========================================================

# ========================= PATHS =========================
INPUT_CSV = f"../{SURVEY}_anonymized.csv"
ITEM_META = f"../../../item_generation/{SURVEY}_combined.json"
OUTPUT_JSON = f"./{SURVEY}_responses.json"
# =========================================================


def load_json(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)

def save_json(data, out_path):
    with open(out_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_likert_map(survey):
    if survey == "big5":
        return {
        "Very Accurate": 5,
        "Moderately Accurate": 4,
        "Neither inaccurate nor accurate": 3,
        "Moderately Inaccurate": 2,
        "Very Inaccurate": 1,
        }   
    elif survey == "pvq":
        return {
        "Very Much Like Me": 6,
        "Like Me": 5,
        "Somewhat Like Me": 4,
        "A Little Like Me": 3,
        "Not Like Me": 2,
        "Not Like Me at All": 1,
        }
    elif "via" in survey:
        return {
        "Very much like me": 5,
        "Like me": 4,
        "Neutral": 3,
        "Unlike me": 2,
        "Very much unlike me": 1,
        }
    else:
        raise ValueError(f"Invalid survey: {survey}")
    return None

def main():
    raw_data = load_json(ITEM_META)
    likert_map = get_likert_map(SURVEY)

    item_meta = {
        entry["q_id"]: {
            "item": entry["item"],
            "trait": entry["expected_trait"],
            "correlation": entry.get("expected_correlation"),
            "source": entry.get("source"),
        }
        for entry in raw_data
    }

    result = {
        q_id: {**meta, "responses": []}
        for q_id, meta in item_meta.items()
    }

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            participant_id = row["Participant_ID"]
            for q_id in item_meta:
                raw = row.get(q_id, "").strip()
                score = likert_map.get(raw)
                if score is not None:
                    result[q_id]["responses"].append([score, participant_id])

    # Save result
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    save_json(result, OUTPUT_JSON)

    print(f"[Done] Output written to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
