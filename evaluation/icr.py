import json
import os
import re
import math

# ===================== CONFIGURATION =====================
SURVEY_TYPES = ['big5', 'pvq', 'via']
SUBSAMPLING_OBJECTS = ['free', 'caps', 'item', 'wvs', 'no_mediator', 'sampling']
NUMS = [500] # [50, 100, 200, 300, 400, 500]
EVAL_RANKS = {
    'big5': [10],
    'pvq': [4],
    'via': [4],
}
# =========================================================

# ========================= PATHS =========================
# Input file paths
INPUT_RANK_DIR = '../item_ranking_and_selection_new/rank/{SUBSAMPLING_OBJECT}/{N_SAMPLES}'
INPUT_RESP_DIR = '../data/human_survey/processed/{SURVEY}_responses.json'
VIA_PARTS_PATH = '../traits_selection/via_parts.json'
VIA_PART1_RESP_PATH = '../data/human_survey/processed/via_part1_responses.json'
VIA_PART2_RESP_PATH = '../data/human_survey/processed/via_part2_responses.json'

# Output file paths
OUTPUT_BASE_DIR = './icr/{SUBSAMPLING_OBJECT}'
OUTPUT_FILE_NAME = 'topN_icr.json'
# Likert max for reverse coding negative items
LIKERT_MAX = {
    'big5': 5,
    'pvq': 6,
    'via': 5,
}
# =========================================================

def load_json(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_rank_path(subsampling_object, n_samples, survey):
    return os.path.join(
        INPUT_RANK_DIR.format(
            SUBSAMPLING_OBJECT=subsampling_object,
            N_SAMPLES=n_samples,
        ),
        f'{survey}_rank_results.json'
    )


def get_response_path(survey):
    return INPUT_RESP_DIR.format(SURVEY=survey)


def load_response_data_by_survey(survey):
    if survey != 'via':
        return load_json(get_response_path(survey))

    via_parts = load_json(VIA_PARTS_PATH)
    trait_to_part = {}
    for trait in via_parts.get('via_part1', {}):
        trait_to_part[trait] = 'via_part1'
    for trait in via_parts.get('via_part2', {}):
        trait_to_part[trait] = 'via_part2'

    return {
        'trait_to_part': trait_to_part,
        'via_part1': load_json(VIA_PART1_RESP_PATH),
        'via_part2': load_json(VIA_PART2_RESP_PATH),
    }


def extract_sample_names(rank_rows, subsampling_object):
    """
    Extract sample names from keys like:
    - caps_sample_001_rank
    - item_sample_127_rank
    """
    pattern = re.compile(rf'^{re.escape(subsampling_object)}_sample_\d{{3}}_rank$')
    sample_names = set()
    for row in rank_rows:
        for key in row.keys():
            if pattern.match(key):
                sample_names.add(key[:-5])  # strip "_rank"
    return sorted(sample_names)


def get_response_item(response_data, qid, expected_trait, survey):
    """Get response item record for (survey, qid, expected_trait)."""
    if survey == 'via':
        part_name = response_data.get('trait_to_part', {}).get(expected_trait)
        if part_name not in ('via_part1', 'via_part2'):
            return None
        part_resp_data = response_data.get(part_name, {})
        return part_resp_data.get(qid)
    return response_data.get(qid)


def variance_sample(values):
    n = len(values)
    if n < 2:
        return None
    mean_v = sum(values) / n
    return sum((v - mean_v) ** 2 for v in values) / (n - 1)


def cronbach_alpha(item_pid_scores):
    """
    item_pid_scores: list of dict(pid -> score), one dict per item
    """
    n_items = len(item_pid_scores)
    if n_items < 2:
        return None

    shared_pids = set(item_pid_scores[0].keys())
    for pid_scores in item_pid_scores[1:]:
        shared_pids &= set(pid_scores.keys())
    shared_pids = sorted(shared_pids)

    if len(shared_pids) < 2:
        return None

    item_variances = []
    for pid_scores in item_pid_scores:
        vals = [pid_scores[pid] for pid in shared_pids]
        var_i = variance_sample(vals)
        if var_i is None:
            return None
        item_variances.append(var_i)

    total_scores = []
    for pid in shared_pids:
        total_scores.append(sum(pid_scores[pid] for pid_scores in item_pid_scores))
    total_var = variance_sample(total_scores)
    if total_var is None or math.isclose(total_var, 0.0):
        return None

    return (n_items / (n_items - 1)) * (1 - (sum(item_variances) / total_var))


def build_item_pid_scores(response_data, qid, expected_trait, survey):
    item = get_response_item(response_data, qid, expected_trait, survey)
    if not item:
        return None

    corr_dir = item.get('correlation')
    pid_scores = {}
    for score, pid in item.get('responses', []):
        coded = (LIKERT_MAX[survey] + 1 - score) if corr_dir == 'negative' else score
        pid_scores[pid] = coded

    if not pid_scores:
        return None
    return pid_scores

def compute_topk_icr_for_survey(rank_rows, response_data, sample_name, top_k, survey):
    """
    Select top-k items per sample and report survey-level mean of trait-wise
    Cronbach alpha (ICR_t).
    """
    rank_key = f'{sample_name}_rank'
    selected_by_trait = {}

    for row in rank_rows:
        rank_value = row.get(rank_key)
        if isinstance(rank_value, int) and 0 < rank_value <= top_k:
            qid = row.get('question_id')
            expected_trait = row.get('expected_trait')
            if qid is not None and expected_trait is not None:
                selected_by_trait.setdefault(expected_trait, []).append(qid)

    trait_alphas = []
    for trait, qids in selected_by_trait.items():
        item_pid_scores = []
        for qid in qids:
            pid_scores = build_item_pid_scores(response_data, qid, trait, survey)
            if pid_scores is not None:
                item_pid_scores.append(pid_scores)

        alpha = cronbach_alpha(item_pid_scores)
        if alpha is not None:
            trait_alphas.append(alpha)

    if not trait_alphas:
        return None
    return sum(trait_alphas) / len(trait_alphas)


def build_results_for_object(subsampling_object):
    """
    Output format:
    {
        "500": {
            "@1": {
                "caps_sample_001": {"big5": 0.67, "pvq": 0.35, "via": 0.49}
            }
        }
    }
    """
    result = {}

    for num in NUMS:
        num_key = str(num)
        result[num_key] = {}

        # load all data first (survey-wise)
        rank_by_survey = {}
        response_by_survey = {}
        samples_by_survey = {}

        for survey in SURVEY_TYPES:
            rank_path = get_rank_path(subsampling_object, num, survey)
            rank_rows = load_json(rank_path)
            response_data = load_response_data_by_survey(survey)

            rank_by_survey[survey] = rank_rows
            response_by_survey[survey] = response_data
            samples_by_survey[survey] = extract_sample_names(rank_rows, subsampling_object)

        # rank columns should be aligned across surveys; use intersection for safety
        common_samples = set(samples_by_survey[SURVEY_TYPES[0]])
        for survey in SURVEY_TYPES[1:]:
            common_samples &= set(samples_by_survey[survey])
        common_samples = sorted(common_samples)

        for survey in SURVEY_TYPES:
            for k in EVAL_RANKS[survey]:
                topk_key = f'@{k}'
                if topk_key not in result[num_key]:
                    result[num_key][topk_key] = {}

                for sample_name in common_samples:
                    if sample_name not in result[num_key][topk_key]:
                        result[num_key][topk_key][sample_name] = {}

                    mean_icr = compute_topk_icr_for_survey(
                        rank_rows=rank_by_survey[survey],
                        response_data=response_by_survey[survey],
                        sample_name=sample_name,
                        top_k=k,
                        survey=survey,
                    )
                    result[num_key][topk_key][sample_name][survey] = mean_icr

    return result


def main():
    for subsampling_object in SUBSAMPLING_OBJECTS:
        result = build_results_for_object(subsampling_object)
        out_path = os.path.join(
            OUTPUT_BASE_DIR.format(SUBSAMPLING_OBJECT=subsampling_object),
            OUTPUT_FILE_NAME
        )
        save_json(result, out_path)
        print(f'[Done] {subsampling_object}: {out_path}')


if __name__ == '__main__':
    main()