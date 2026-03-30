import os
import json
import math
import pandas as pd
import re
from multiprocessing import Pool, cpu_count
import time

# ===================== CONFIGURATION =====================
# Multiprocessing settings
MAX_PROCESSES = None  # None = auto-detect using CPU_USAGE_RATIO, or set specific number
CPU_USAGE_RATIO = 0.75  # Use 75% of available CPU cores (0.0 to 1.0)
                        # Examples: 0.5 = 50%, 0.75 = 75%, 1.0 = 100%
                        # Recommended: 0.75 to leave some cores for system processes
CHUNK_SIZE = 1

# config
SUBSAMPLING_OBJECT = "free"  # free, caps, wvs, item, no_mediator, sampling
SURVEYS = ['big5', 'pvq', 'via']
NUMS = [500]  # [50, 100, 200, 300, 400, 500]
# =========================================================


# ========================= PATHS =========================
# Input file paths
INPUT_BASE_DIR   = './subsampling_cv/{SUBSAMPLING_OBJECT}/{N_SAMPLES}/sample_{sample_idx:03d}'
ITEM_FILE_PATH   = '../item_generation/{survey_type}_combined.json'
CORR_FILE_PATH   = '{survey_type}_cv_results.json'

# Output file paths
OUTPUT_BASE_DIR  = './rank/{SUBSAMPLING_OBJECT}/{N_SAMPLES}'
OUTPUT_FILE_PATTERN = '{survey_type}_rank_results.json'
# =========================================================


def calculate_process_count(cpu_ratio=None, max_processes=None):
    total_cores = cpu_count()
    if max_processes is not None:
        return min(max_processes, total_cores)
    elif cpu_ratio is not None:
        return max(1, int(total_cores * cpu_ratio))
    else:
        # default: 75% usage
        return max(1, int(total_cores * 0.75))

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_trait(survey):
    trait_file = "../traits_selection/trait.json"
    trait = load_json(trait_file)
    return list(trait[survey].keys())

def load_items(survey):
    """Load items from ITEM_FILE_PATH (combined.json), excluding bogus."""
    if survey == 'via':
        part1 = load_json(ITEM_FILE_PATH.format(survey_type='via_part1'))
        part2 = load_json(ITEM_FILE_PATH.format(survey_type='via_part2'))
        items = part1 + part2
    else:
        items = load_json(ITEM_FILE_PATH.format(survey_type=survey))
    return [it for it in items if it.get('expected_trait') != 'bogus']

def get_source_key(source):
    """Map item source name to the key used in correlation_results."""
    if source == 'psy':
        return 'original'
    elif source == 'llama3.1-8b, llama3.3-70b':
        return 'llama3.1-8b'
    return source

# Some subsampling objects use a different key inside the correlations dict
CORR_KEY_MAP = {
    'item': 'M_item',
}

def get_cv_correlation(item_text, trait, source_key, rank_validity):
    """
    Look up the CV correlation for an item from rank_validity.
    Returns the correlation value for SUBSAMPLING_OBJECT, or None if not found.
    """
    corr_key = CORR_KEY_MAP.get(SUBSAMPLING_OBJECT, SUBSAMPLING_OBJECT)
    source_entries = rank_validity.get(source_key)
    if source_entries is None:
        return None
    for entry in source_entries:
        if entry['item'] == item_text and entry['trait'] == trait:
            corr_info = entry.get('correlations', {}).get(corr_key)
            if corr_info is not None:
                return corr_info['correlation']
    return None

def tie_break(df, traits, ranking_columns):
    """
    Tie breaking logic (from 2_tie_break.py).

    Priority:
      1. original rank value (ascending)
      2. generated_number (ascending - earlier generated items preferred)
      3. item text length (ascending - shorter items preferred)
      4. item text (ascending - alphabetical as final fallback)
    """
    df_result = df.copy()

    if 'item_length' not in df_result.columns:
        df_result['item_length'] = df_result['item'].str.len()

    for t in traits:
        trait_mask = (df_result["expected_trait"] == t) & (df_result["source"] != "psy")
        if not trait_mask.any():
            continue

        for col in ranking_columns:
            trait_indices = df_result.index[trait_mask]
            tie_break_cols = [col, "generated_number", "item_length", "item"]
            trait_data = df_result.loc[trait_indices, tie_break_cols].copy()

            rank_counts = trait_data[col].value_counts()
            tied_ranks = rank_counts[rank_counts > 1].index

            if len(tied_ranks) > 0:
                trait_data_sorted = trait_data.sort_values(
                    [col, "generated_number", "item_length", "item"],
                    ascending=[True, True, True, True]
                )
                new_ranks = pd.Series(
                    range(1, len(trait_data_sorted) + 1),
                    index=trait_data_sorted.index
                )
                df_result.loc[trait_indices, col] = new_ranks

                # Final safety: index-based tie breaking
                remaining_ties = new_ranks.value_counts()
                remaining_ties = remaining_ties[remaining_ties > 1]
                if len(remaining_ties) > 0:
                    trait_data_final = df_result.loc[trait_indices, [col]].copy()
                    trait_data_final['original_index'] = trait_data_final.index
                    trait_data_final_sorted = trait_data_final.sort_values([col, 'original_index'])
                    final_ranks = pd.Series(
                        range(1, len(trait_data_final_sorted) + 1),
                        index=trait_data_final_sorted.index
                    )
                    df_result.loc[trait_indices, col] = final_ranks

    if 'item_length' in df_result.columns:
        df_result = df_result.drop('item_length', axis=1)

    return df_result


def process_single_survey_num(task_args):
    SURVEY, NUM = task_args

    try:
        print(f"Processing {SURVEY} - {NUM} samples (PID: {os.getpid()})...")

        trait = get_trait(SURVEY)

        # ── 1. Load items from ITEM_FILE_PATH ──────────────────────────────
        items = load_items(SURVEY)

        # Use (q_id, item) as internal key to avoid q_id collisions in via
        # (via_part1 and via_part2 share the same q_id namespace)
        base_records = {}
        for it in items:
            key = (it['q_id'], it['item'])
            base_records[key] = {
                'question_id':          it['q_id'],
                'item':                 it['item'],
                'expected_trait':       it['expected_trait'],
                'expected_correlation': it['expected_correlation'],
                'source':               it['source'],
                'generated_number':     it.get('generated_number', 0),
            }

        # ── 2. Load CV correlation for each sample ──────────────────────────
        max_idx   = 1 if NUM == 500 else 1001
        sample_columns = []

        for idx in range(1, max_idx + 1):
            corr_path = os.path.join(
                INPUT_BASE_DIR.format(
                    SUBSAMPLING_OBJECT=SUBSAMPLING_OBJECT,
                    N_SAMPLES=NUM,
                    sample_idx=idx
                ),
                CORR_FILE_PATH.format(survey_type=SURVEY.lower())
            )

            if not os.path.exists(corr_path):
                print(f"  Warning: {corr_path} not found, skipping.")
                continue

            rank_validity = load_json(corr_path)
            col_name = f'{SUBSAMPLING_OBJECT}_sample_{idx:03d}'
            sample_columns.append(col_name)

            for key, record in base_records.items():
                source_key = get_source_key(record['source'])
                cv_corr = get_cv_correlation(
                    record['item'],
                    record['expected_trait'],
                    source_key,
                    rank_validity
                )
                # Normalize null/NaN to None
                if cv_corr is not None:
                    try:
                        if math.isnan(cv_corr):
                            cv_corr = None
                    except TypeError:
                        cv_corr = None

                # Sign adjustment for negative items
                if cv_corr is not None and record['expected_correlation'] == 'negative':
                    cv_corr = cv_corr * -1
                record[col_name] = cv_corr

        # ── 3. Build DataFrame ──────────────────────────────────────────────
        df = pd.DataFrame(list(base_records.values()))

        # ── 4. Initial rank per trait per sample (non-psy only) ────────────
        # null corr items (and psy items) are excluded from ranking → rank = -1.
        rank_columns = []
        for col in sample_columns:
            rank_col = f'{col}_rank'
            rank_columns.append(rank_col)
            for t in trait:
                mask = (df["expected_trait"] == t) & (df["source"] != 'psy') & df[col].notna()
                df.loc[mask, rank_col] = (
                    df[mask][col]
                    .rank(method='min', ascending=False)
                    .astype(int)
                )
            # psy items and null corr items: rank = -1
            df[rank_col] = df[rank_col].fillna(-1).astype(int)

        # ── 5. Tie breaking ────────────────────────────────────────────────
        df = tie_break(df, trait, rank_columns)

        # Ensure rank columns are integers after tie breaking
        df[rank_columns] = df[rank_columns].apply(lambda x: x.fillna(-1).astype(int))

        # ── 6. Enforce output column order ─────────────────────────────────
        keep_columns = [
            'question_id', 'item', 'expected_trait',
            'expected_correlation', 'source', 'generated_number'
        ]
        ordered_columns = keep_columns.copy()
        for col in sample_columns:
            ordered_columns.append(col)
            ordered_columns.append(f'{col}_rank')

        df = df[[c for c in ordered_columns if c in df.columns]]

        # ── 7. Save ────────────────────────────────────────────────────────
        output_dir = OUTPUT_BASE_DIR.format(
            SUBSAMPLING_OBJECT=SUBSAMPLING_OBJECT, N_SAMPLES=NUM
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            OUTPUT_FILE_PATTERN.format(survey_type=SURVEY.lower())
        )
        save_json(df.to_dict(orient='records'), output_file)
        print(f"  Results saved to {output_file}")

        return f"Success: {SURVEY} - {NUM} samples"

    except Exception as e:
        import traceback
        error_msg = f"Error: {SURVEY} - {NUM} samples: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def main():
    tasks = [(SURVEY, NUM) for SURVEY in SURVEYS for NUM in NUMS]

    print(f"Total {len(tasks)} tasks to process.")
    print(f"Available CPU cores: {cpu_count()}")

    num_processes = calculate_process_count(CPU_USAGE_RATIO, MAX_PROCESSES)
    print(f"Using {num_processes} processes ({num_processes/cpu_count()*100:.1f}% of {cpu_count()} cores)")
    print(f"Surveys: {SURVEYS}")
    print(f"Nums: {NUMS}")

    start_time = time.time()

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_survey_num, tasks, chunksize=CHUNK_SIZE)

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r.startswith("Success"))
    error_count   = len(results) - success_count

    print(f"\n=== Processing completed ===")
    print(f"Total time: {elapsed:.2f}s  |  Success: {success_count}  |  Failure: {error_count}")

    if error_count > 0:
        print("\nFailed tasks:")
        for r in results:
            if r.startswith("Error"):
                print(f"  {r}")


if __name__ == "__main__":
    main()
