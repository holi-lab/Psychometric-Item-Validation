import json
import os
from scipy import stats

# ===================== CONFIGURATION =====================
SURVEY = "pvq"  # big5, pvq, via_part1, via_part2
# =========================================================

# ========================= PATHS =========================
INPUT_JSON  = f"../processed/{SURVEY}_responses.json"
OUTPUT_JSON = f"./{SURVEY}_spearman_correlations.json"
# =========================================================

# ========================= CONSTANTS =========================
LIKERT_MAX = {  # maximum score on the Likert scale (used for reverse coding)
    "big5": 5,
    "pvq": 6,
    "via_part1": 5,
    "via_part2": 5,
    # "via": 5,
}
# =========================================================


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def significance_label(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def build_participant_item_scores(responses_data):
    """Return {q_id: {participant_id: score}} from the responses JSON."""
    item_scores = {}
    for q_id, entry in responses_data.items():
        item_scores[q_id] = {pid: score for score, pid in entry["responses"]}
    return item_scores


def build_trait_scores(responses_data, item_scores):
    """
    For each trait, compute per-participant mean scores across psy-source items only,
    applying reverse coding (LIKERT_MAX[SURVEY] + 1 - score) for items
    whose expected_correlation is 'negative'.
    For pvq, additionally subtracts each participant's grand mean across all psy items
    (ipsatization) to control for acquiescence bias.
    Returns {trait: {participant_id: trait_score}}.
    """
    trait_items = {}
    for q_id, entry in responses_data.items():
        if entry.get("source") != "psy":
            continue
        if entry.get("trait") == "bogus":
            continue
        trait = entry["trait"]
        trait_items.setdefault(trait, []).append(q_id)

    all_participants = {pid for scores in item_scores.values() for pid in scores}

    # pvq: 참여자별 모든 psy 문항 응답의 grand mean 계산 (bogus 제외)
    if SURVEY == "pvq":
        all_psy_qids    = [q for q_ids in trait_items.values() for q in q_ids]
        psy_participants = {pid for q_id in all_psy_qids for pid in item_scores[q_id]}
        grand_sum   = {pid: 0 for pid in psy_participants}
        grand_count = {pid: 0 for pid in psy_participants}
        for q_id in all_psy_qids:
            corr_dir = responses_data[q_id].get("correlation")
            for pid, score in item_scores[q_id].items():
                coded = (LIKERT_MAX[SURVEY] + 1 - score) if corr_dir == "negative" else score
                grand_sum[pid]   += coded
                grand_count[pid] += 1
        grand_mean = {
            pid: grand_sum[pid] / grand_count[pid]
            for pid in psy_participants
            if grand_count[pid] > 0
        }

    trait_scores = {}
    for trait, q_ids in trait_items.items():
        sum_scores = {pid: 0 for pid in all_participants}
        counts     = {pid: 0 for pid in all_participants}
        for q_id in q_ids:
            corr_dir = responses_data[q_id].get("correlation")
            for pid, score in item_scores[q_id].items():
                coded = (LIKERT_MAX[SURVEY] + 1 - score) if corr_dir == "negative" else score
                sum_scores[pid] += coded
                counts[pid] += 1
        trait_scores[trait] = {
            pid: (sum_scores[pid] / counts[pid]) - (grand_mean[pid] if SURVEY == "pvq" else 0)
            for pid in all_participants
            if counts[pid] > 0
        }

    return trait_scores


def compute_spearman_for_item(item_scores_dict, trait_composites):
    """
    Compute Spearman correlation between one item's scores and each trait
    composite score, using only participants present in both vectors.
    Returns {trait: {correlation_value, p_value, significance}}.
    """
    correlations = {}
    item_pids = set(item_scores_dict.keys())

    for trait, composite in trait_composites.items():
        shared_pids = sorted(item_pids & set(composite.keys()))
        if len(shared_pids) < 3:
            correlations[trait] = {
                "correlation_value": None,
                "p_value": None,
                "significance": "",
            }
            continue

        x = [item_scores_dict[pid] for pid in shared_pids]
        y = [composite[pid] for pid in shared_pids]

        rho, p_val = stats.spearmanr(x, y)
        correlations[trait] = {
            "correlation_value": float(rho),
            "p_value": float(p_val),
            "significance": significance_label(p_val),
        }

    return correlations


def main():
    responses_data = load_json(INPUT_JSON)
    item_scores    = build_participant_item_scores(responses_data)
    trait_composites = build_trait_scores(responses_data, item_scores)

    result = {}
    for q_id, entry in responses_data.items():
        if entry.get("trait") == "bogus":
            continue
        correlations = compute_spearman_for_item(item_scores[q_id], trait_composites)
        result[q_id] = {
            "item":                 entry["item"],
            "expected_trait":       entry["trait"],
            "expected_correlation": entry.get("correlation"),
            "source":               entry.get("source"),
            "correlations":         correlations,
        }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    save_json(result, OUTPUT_JSON)
    print(f"[Done] Output written to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
