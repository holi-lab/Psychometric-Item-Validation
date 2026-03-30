import json
import os
import re

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
INPUT_CORR_DIR = '../data/human_survey/spearman_correlation_new/{SURVEY}_spearman_correlations.json'
VIA_PARTS_PATH = '../traits_selection/via_parts.json'
VIA_PART1_CORR_PATH = '../data/human_survey/spearman_correlation/via_part1_spearman_correlations.json'
VIA_PART2_CORR_PATH = '../data/human_survey/spearman_correlation/via_part2_spearman_correlations.json'

# Output file paths
OUTPUT_BASE_DIR = './cv/{SUBSAMPLING_OBJECT}'
OUTPUT_FILE_NAME = 'topN_cv.json'
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


def get_corr_path(survey):
    return INPUT_CORR_DIR.format(SURVEY=survey)


def load_corr_data_by_survey(survey):
    if survey != 'via':
        return load_json(get_corr_path(survey))

    via_parts = load_json(VIA_PARTS_PATH)
    trait_to_part = {}
    for trait in via_parts.get('via_part1', {}):
        trait_to_part[trait] = 'via_part1'
    for trait in via_parts.get('via_part2', {}):
        trait_to_part[trait] = 'via_part2'

    return {
        'trait_to_part': trait_to_part,
        'via_part1': load_json(VIA_PART1_CORR_PATH),
        'via_part2': load_json(VIA_PART2_CORR_PATH),
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


def get_expected_trait_corr_abs(corr_data, qid, expected_trait, survey):
    """
    Return abs(correlation) for expected trait of qid.
    Uses expected_correlation direction to flip sign first, then abs.
    """
    if survey == 'via':
        part_name = corr_data.get('trait_to_part', {}).get(expected_trait)
        if part_name not in ('via_part1', 'via_part2'):
            return None
        part_corr_data = corr_data.get(part_name, {})
        item = part_corr_data.get(qid)
    else:
        item = corr_data.get(qid)

    if not item:
        return None

    expected_dir = item.get('expected_correlation')
    corr_value = (
        item.get('correlations', {})
        .get(expected_trait, {})
        .get('correlation_value')
    )

    if corr_value is None:
        return None

    adjusted = -corr_value if expected_dir == 'negative' else corr_value
    return abs(adjusted)


def compute_topk_mean_for_survey(rank_rows, corr_data, sample_name, top_k, survey):
    """
    Pick items with 0 < sample_rank <= top_k and compute mean abs corr.
    """
    rank_key = f'{sample_name}_rank'
    selected_items = []

    for row in rank_rows:
        rank_value = row.get(rank_key)
        if isinstance(rank_value, int) and 0 < rank_value <= top_k:
            qid = row.get('question_id')
            expected_trait = row.get('expected_trait')
            if qid is not None and expected_trait is not None:
                selected_items.append((qid, expected_trait))

    values = []
    for qid, expected_trait in selected_items:
        v = get_expected_trait_corr_abs(corr_data, qid, expected_trait, survey)
        if v is not None:
            values.append(v)

    if not values:
        return None
    return sum(values) / len(values)


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
        corr_by_survey = {}
        samples_by_survey = {}

        for survey in SURVEY_TYPES:
            rank_path = get_rank_path(subsampling_object, num, survey)
            rank_rows = load_json(rank_path)
            corr_data = load_corr_data_by_survey(survey)

            rank_by_survey[survey] = rank_rows
            corr_by_survey[survey] = corr_data
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

                    mean_corr = compute_topk_mean_for_survey(
                        rank_rows=rank_by_survey[survey],
                        corr_data=corr_by_survey[survey],
                        sample_name=sample_name,
                        top_k=k,
                        survey=survey,
                    )
                    result[num_key][topk_key][sample_name][survey] = mean_corr

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