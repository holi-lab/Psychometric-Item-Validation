import json
import pandas as pd
from scipy import stats
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# ===================== CONFIGURATION =====================
# Multiprocessing settings
MAX_PROCESSES = None  # None = auto-detect using CPU_USAGE_RATIO, or set specific number
CPU_USAGE_RATIO = 0.75  # Use 75% of available CPU cores (0.0 to 1.0)
                        # Examples: 0.5 = 50%, 0.75 = 75%, 1.0 = 100%
                        # Recommended: 0.75 to leave some cores for system processes
CHUNK_SIZE = 1  # Number of tasks per chunk for pool.map

# Models to analyze (item generation models)
MODELS = ['original', 'gpt-4o', 'gpt-4o-mini', 'llama3.1-8b', 'llama3.3-70b']

# Survey types to process
SURVEY_TYPES = ['big5', 'pvq', 'via']

# Subsampling object & prompt types to analyze
SUBSAMPLING_OBJECT = 'free' # free, caps, wvs, item, no_mediator, sampling
PROMPT_TYPES = ['free'] # ['free', 'caps', 'wvs', 'M_item', 'no_mediator', 'sampling']

# PC IDs to process
def sample_numbers(n, replacement=False):
    """replacement=True -> Bootstrap
       replacement=False -> Subsampling"""
    np.random.seed(42 + n)
    
    if n == 500: # 500 is the number of full data
        return {1: list(range(1, 501))}
    
    result = {}
    for i in range(1, 1002):
        result[i] = np.random.choice(np.arange(1, 501), size=n, replace=replacement).tolist()
    return result

# Subsampling object & settings
N_SAMPLES = 500
REPLACEMENT = False
SAMPLE_SETS = sample_numbers(N_SAMPLES, replacement=REPLACEMENT)
# =========================================================

# ========================= PATHS =========================
# Input/Output paths
INPUT_BASE_DIR = f'../simulation_results/{SUBSAMPLING_OBJECT}/processed'
OUTPUT_BASE_DIR = f'./subsampling_cv/{SUBSAMPLING_OBJECT}/{N_SAMPLES}'

# File naming patterns
SCORES_FILE_PATTERN = '{survey_type}_processed_scores.json'
PROMPTS_FILE_PATTERN = '{survey_type}_{model}_prompts.json'
OUTPUT_FILE_PATTERN = '{survey_type}_cv_results.json'
# =========================================================

def calculate_process_count(cpu_ratio=None, max_processes=None):
    """
    Calculate the number of processes to use based on CPU usage or maximum number of processes.
    
    Args:
        cpu_ratio (float): CPU usage (0.0 ~ 1.0)
        max_processes (int): maximum number of processes
    
    Returns:
        int: number of processes to use
    """
    total_cores = cpu_count()
    
    if max_processes is not None:
        return min(max_processes, total_cores)
    elif cpu_ratio is not None:
        return max(1, int(total_cores * cpu_ratio))
    else:
        # default: 75% usage
        return max(1, int(total_cores * 0.75))

def load_json(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)

def calculate_correlations(prompts_data, scores_data):
    # Group prompts data by item
    item_groups = {}
    for entry in prompts_data:
        item = entry['item']
        if item not in item_groups:
            item_groups[item] = []
        item_groups[item].append(entry)
    
    results = []
    
    # Calculate correlations for each item
    for item, entries in item_groups.items():
        # Get the first entry to extract trait information
        first_entry = entries[0]
        trait = first_entry['trait']
        
        # Collect responses for all person_ids
        person_responses = {}
        for entry in entries:
            person_id = entry['person_id']
            
            try:
                # Create response dictionary with model output
                responses = {
                    prompt_type: entry[prompt_type]
                    for prompt_type in PROMPT_TYPES
                }
                
                # Add actual values for each prompt type
                if person_id in scores_data and trait in scores_data[person_id]:
                    trait_data = scores_data[person_id][trait]
                    prompt_means = {
                        prompt_type: trait_data[prompt_type]['mean']
                        for prompt_type in PROMPT_TYPES
                    }
                    responses.update({f'{k}_actual': v for k, v in prompt_means.items()})
                    person_responses[person_id] = responses
            except Exception as e:
                print(f"Error processing person_id {person_id} for item {item}: {str(e)}")
                continue
        
        # Calculate correlations
        if person_responses:
            df = pd.DataFrame.from_dict(person_responses, orient='index')
            correlations = {}
            
            # Print debug information
            print(f"\nDebug info for item: {item}")
            print(f"Number of responses: {len(df)}")
            
            # Process each prompt type
            for prompt_type in PROMPT_TYPES:
                actual_col = f'{prompt_type}_actual'
                
                if actual_col not in df.columns or prompt_type not in df.columns:
                    print(f"Warning: Missing data for {prompt_type}")
                    continue
                
                try:
                    # Check if we have enough unique values
                    actual_unique = len(df[actual_col].unique())
                    response_unique = len(df[prompt_type].unique())
                    
                    print(f"\n{prompt_type}:")
                    print(f"Sample of actual values ({actual_col}):", df[actual_col].head().tolist())
                    print(f"Sample of responses ({prompt_type}):", df[prompt_type].head().tolist())
                    
                    if actual_unique <= 1 or response_unique <= 1:
                        print(f"Warning: Constant values detected for {prompt_type} in item {item}")
                        print(f"Unique values - actual: {actual_unique}, response: {response_unique}")
                        print("Actual values:", sorted(df[actual_col].unique().tolist()))
                        print("Response values:", sorted(df[prompt_type].unique().tolist()))
                        correlations[prompt_type] = {
                            'correlation': None,
                            'p_value': None,
                            'actual_unique_values': actual_unique,
                            'response_unique_values': response_unique,
                            'actual_values': sorted(df[actual_col].unique().tolist()),
                            'response_values': sorted(df[prompt_type].unique().tolist())
                        }
                    else:
                        correlation, p_value = stats.spearmanr(df[actual_col], df[prompt_type])
                        correlations[prompt_type] = {
                            'correlation': float(correlation) if not pd.isna(correlation) else None,
                            'p_value': float(p_value) if not pd.isna(p_value) else None,
                            'actual_unique_values': actual_unique,
                            'response_unique_values': response_unique,
                            'actual_values': sorted(df[actual_col].unique().tolist()),
                            'response_values': sorted(df[prompt_type].unique().tolist())
                        }
                except Exception as e:
                    print(f"Error calculating correlation for {prompt_type} in item {item}: {str(e)}")
                    correlations[prompt_type] = {
                        'correlation': None,
                        'p_value': None,
                        'error': str(e)
                    }
            
            result = {
                'item': item,
                'trait': trait,
                'correlations': correlations,
                'n_responses': len(person_responses)
            }
            results.append(result)
    
    return results

def process_survey_type(survey_type, models, pc_ids, sample_idx):
    print(f"\nProcessing {survey_type} survey type for PC range {len(pc_ids)}...")
    
    # Load scores data
    scores_file = os.path.join(INPUT_BASE_DIR, SCORES_FILE_PATTERN.format(survey_type=survey_type.lower()))
    print(f"Loading scores from {scores_file}")
    scores_data = load_json(scores_file)
    
    # Store all results
    all_results = {}
    
    # Process each model
    for model in models:
        filename = os.path.join(INPUT_BASE_DIR, PROMPTS_FILE_PATTERN.format(
            survey_type=survey_type.lower(),
            model=model
        ))
        print(f"\nProcessing {filename}...")
        
        try:
            model_data = load_json(filename)
            # Filter data by PC range
            results = calculate_correlations(model_data, scores_data)
            
            # Store results
            all_results[model] = results
            print(f"Successfully processed {len(results)} items for model {model}")
        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")
            continue
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(OUTPUT_BASE_DIR, f'sample_{sample_idx:03d}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, OUTPUT_FILE_PATTERN.format(
        survey_type=survey_type.lower(),
        sample_idx=sample_idx
    ))
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def process_single_task(task_args):
    """
    단일 작업을 처리하는 함수 (멀티프로세싱용)
    task_args: (survey_type, sample_idx, pc_list)
    """
    survey_type, sample_idx, pc_list = task_args
    
    try:
        print(f"Processing {survey_type} - sample {sample_idx:03d} (PID: {os.getpid()})")
        process_survey_type(survey_type, MODELS, pc_list, sample_idx)
        return f"Success: {survey_type} - sample {sample_idx:03d}"
    except Exception as e:
        error_msg = f"Error: {survey_type} - sample {sample_idx:03d}: {str(e)}"
        print(error_msg)
        return error_msg

def main():
    # Create tasks list
    tasks = []
    for survey_type in SURVEY_TYPES:
        for s_idx, pc_list in SAMPLE_SETS.items():
            tasks.append((survey_type, s_idx, pc_list))
    
    print(f"Total {len(tasks)} tasks to process.")
    print(f"Available CPU cores: {cpu_count()}")
    
    # Set number of processes
    num_processes = calculate_process_count(CPU_USAGE_RATIO, MAX_PROCESSES)
    
    print(f"Number of processes to use: {num_processes} (total {cpu_count()} cores, {num_processes/cpu_count()*100:.1f}% usage)")
    print(f"Survey types: {SURVEY_TYPES}")
    print(f"Sample sets: {len(SAMPLE_SETS)}")
    
    start_time = time.time()
    
    # Run multiprocessing
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_task, tasks, chunksize=CHUNK_SIZE)
    
    end_time = time.time()
    
    # Summary of results
    success_count = sum(1 for result in results if result.startswith("Success"))
    error_count = len(results) - success_count
    
    print(f"\n=== Processing completed ===")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Average time per task: {(end_time - start_time) / len(tasks):.2f} seconds")
    print(f"Success: {success_count} tasks")
    print(f"Failure: {error_count} tasks")
    
    if error_count > 0:
        print("\nFailed tasks:")
        for result in results:
            if result.startswith("Error"):
                print(f"  {result}")

if __name__ == "__main__":
    main() 