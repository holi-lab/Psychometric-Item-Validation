import json

# ========================= PATHS =========================
PART1_JSON  = f"./via_part1_spearman_correlations.json"
PART2_JSON  = f"./via_part2_spearman_correlations.json"
OUTPUT_JSON = f"./via_spearman_correlations.json"
# =========================================================


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    part1 = load_json(PART1_JSON)
    part2 = load_json(PART2_JSON)
    result = {
        "via_part1": part1,
        "via_part2": part2,
    }
    save_json(result, OUTPUT_JSON)
    print(f"[Done] Output written to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
