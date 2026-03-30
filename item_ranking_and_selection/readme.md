## subsampling_cv: `{survey}_cv_results.json`

| Field | Description |
|---|---|
| Top-level key (e.g., `"gpt-4o"`) | Item generation model name |
| `item` | Survey item |
| `trait` | Target trait the item measures |
| `n_responses` | Number of participants used for correlation computation |
| `correlations.{prompt_type}.correlation` | Spearman correlation between LLM responses and trait scores (simulation-based) |
| `correlations.{prompt_type}.p_value` | p-value of the Spearman correlation |
| `actual_unique_values` | Number of unique values in mean trait scores |
| `response_unique_values` | Number of unique values in LLM responses |
| `actual_values` | Sorted list of unique mean trait scores |
| `response_values` | Sorted list of unique LLM response values |
| `correlation: null` | Indicates correlation could not be computed (constant values or error) |

## rank: `{survey}_rank_results.json`

| Field | Description |
|---|---|
| `question_id` | Unique identifier for the survey item (e.g., `N1`, `N2`) |
| `item` | Survey item |
| `expected_trait` | Target trait the item is designed to measure |
| `expected_correlation` | Expected response direction relative to the trait (`positive` or `negative`) |
| `source` | Origin of the item: `psy` = official psychometric item, otherwise the LLM model that generated it (e.g., `gpt-4o-mini`) |
| `generated_number` | Index of the item among those generated for the same trait by the same model. `0` for `psy` source items |
| `{method}_sample_{idx}` (e.g., `free_sample_001`) | Mean Spearman correlation across all virtual respondents for this item in the given subsample |
| `{method}_sample_{idx}_rank` (e.g., `free_sample_001_rank`) | Rank of this item within its trait group based on correlation. `-1` for `psy` source items (excluded from ranking) |