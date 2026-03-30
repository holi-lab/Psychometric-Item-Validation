import json
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

load_dotenv()

# ===================== CONFIGURATION =====================
# Output directory for generated prompts
OUTPUT_DIR = "prompts"

# Models (item sources) and trait types to process
MODELS = ["gpt-4o", "gpt-4o-mini", "llama3.1-8b", "llama3.3-70b", "psy"]
TRAIT_TYPES = ["big5", "pvq", "via"]

# Mediator configurations
#   key   : output field name in JSON (also produces {key}_inv)
#   field : persona data field pattern ({trait} is replaced with trait name)
#   mode  : "steering"  -> "I highly value {trait}..." prefix (paper default)
#           "vanilla"   -> no trait steering prefix
#           "oppose"    -> "I oppose value {trait}..." prefix
MEDIATORS = [
    {"key": "free",    "field": "caps_{trait}",    "mode": "steering"},
    {"key": "caps",    "field": "free_{trait}",    "mode": "steering"},
    # {"key": "item",           "field": "item_{trait}",          "mode": "steering"},
    # {"key": "wvs",            "field": "wvs_{trait}",           "mode": "steering"},
    # {"key": "no_mediator",    "field": "baseline",              "mode": "steering"},
    # {"key": "persona",        "field": "baseline",              "mode": "vanilla"},
]
# =========================================================

# Derived paths (script lives in mediator_guided_simulation/)
BASE_DIR = Path(__file__).parent.parent
PERSONA_PATH = BASE_DIR / "mediator_generation" / "Persona+Mediator.json"
TRAIT_PATH = BASE_DIR / "traits_selection" / "trait.json"
ITEM_DIR = BASE_DIR / "item_generation"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Item files per trait type (VIA is split into two parts)
ITEM_FILES = {
    "big5": ["big5_combined.json"],
    "pvq":  ["pvq_combined.json"],
    "via":  ["via_part1_combined.json", "via_part2_combined.json"],
}

# Survey-specific instruction and answer choices (matches the paper exactly)
# Paper ref: appendix.tex Section "Prompt Templates for Mediator-Guided Simulation"
# - Big5:  5-point scale (Very Inaccurate ~ Very Accurate)
# - PVQ:   6-point scale (Not Like Them at All ~ Very Much Like Them)
# - VIA:   5-point scale (Very Much Unlike Them ~ Very Much Like Them)
SURVEY_CONFIG = {
    "big5": {
        "instruction": "Based on all the information provided above, select only one answer from the <Answer Choices> to indicate how accurately the <Statement> describes this person's typical behavior or attitudes.",
        "choices": "[Very Inaccurate, Moderately Inaccurate, Neither inaccurate nor accurate, Moderately Accurate, Very Accurate]",
        "choices_rev": "[Very Accurate, Moderately Accurate, Neither inaccurate nor accurate, Moderately Inaccurate, Very Inaccurate]",
    },
    "pvq": {
        "instruction": "Based on all the information provided above, select only one answer from the <Answer Choices> to indicate the degree to which this person is like them, as described in the <Statement>.",
        "choices": "[Not Like Them at All, Not Like Them, A Little Like Them, Somewhat Like Them, Like Them, Very Much Like Them]",
        "choices_rev": "[Very Much Like Them, Like Them, Somewhat Like Them, A Little Like Them, Not Like Them, Not Like Them at All]",
    },
    "via": {
        "instruction": "Based on all the information provided above, select only one answer from the <Answer Choices> to indicate the degree to which the <Statement> describes what the person is like.",
        "choices": "[Very Much Unlike Them, Unlike Them, Neutral, Like Them, Very Much Like Them]",
        "choices_rev": "[Very Much Like Them, Like Them, Neutral, Unlike Them, Very Much Unlike Them]",
    },
}


def load_data():
    """Load persona and trait data."""
    with open(PERSONA_PATH, 'r', encoding='utf-8') as f:
        persona_data = json.load(f)
        if isinstance(persona_data, dict):
            persona_data = [{**v, "id": k} for k, v in persona_data.items()]

    with open(TRAIT_PATH, 'r', encoding='utf-8') as f:
        trait_data = json.load(f)

    return persona_data, trait_data


def load_items(trait_type, model):
    """Load items from combined JSON files, filtered by source model.

    Returns list of dicts grouped by trait:
        [{"trait": ..., "correlations": {"positive": [...], "negative": [...]}}]
    """
    all_items = []
    for filename in ITEM_FILES[trait_type]:
        filepath = ITEM_DIR / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            all_items.extend(json.load(f))

    # Filter by source model (skip bogus/attention-check items that lack 'source')
    filtered = [it for it in all_items if it.get("source") == model]

    # Group by trait → correlation direction
    grouped = defaultdict(lambda: {"positive": [], "negative": []})
    for it in filtered:
        grouped[it["expected_trait"]][it["expected_correlation"]].append(
            {"item": it["item"], "q_id": it["q_id"]}
        )

    return [
        {"trait": trait, "correlations": corrs}
        for trait, corrs in grouped.items()
    ]


def create_prompt(persona_info, trait, trait_def, item, trait_type, mode="steering"):
    """Generate (prompt, prompt_inverted) pair matching the paper's format.

    Paper format (main.tex L256, appendix.tex L287-340):
        I highly value {trait}, which means '{trait_def}'.
        {mediator-integrated persona profile}
        <Instruction> ...
        <Statement> ...
        <Answer Choices> ...
    """
    cfg = SURVEY_CONFIG[trait_type]
    instruction = cfg["instruction"]
    choices = cfg["choices"]
    choices_rev = cfg["choices_rev"]

    if mode == "steering":
        preamble = f"I highly value {trait}, which means '{trait_def}'.\n    {persona_info}"
    elif mode == "oppose":
        preamble = f"I oppose value {trait}, which means '{trait_def}'.\n    {persona_info}"
    else:  # vanilla
        preamble = persona_info

    body = (f"\n    <Instruction>\n"
            f"    {instruction}\n"
            f"    <Statement>\n"
            f"    {item}\n"
            f"    <Answer Choices>\n    ")

    return preamble + body + choices, preamble + body + choices_rev


def process_single_person(args):
    person, questions, trait_data, trait_type, model_name = args
    output_data = []

    try:
        for trait_questions in questions:
            trait = trait_questions["trait"]
            trait_def = trait_data[trait_type][trait]

            # Resolve all mediator fields for this trait
            mediator_texts = {}
            skip = False
            for med in MEDIATORS:
                field = med["field"].replace("{trait}", trait)
                text = person.get(field)
                if not text:
                    skip = True
                    break
                mediator_texts[med["key"]] = (text, med["mode"])
            if skip:
                continue

            for side in trait_questions["correlations"]:
                for item_data in trait_questions["correlations"][side]:
                    entry = {
                        "person_id": person["id"],
                        "trait": trait,
                        "positive_negative": side,
                        "item": item_data["item"],
                        "q_id": item_data["q_id"],
                    }
                    for med_key, (med_text, med_mode) in mediator_texts.items():
                        p, p_inv = create_prompt(
                            med_text, trait, trait_def,
                            item_data["item"], trait_type, mode=med_mode
                        )
                        entry[med_key] = p
                        entry[f"{med_key}_inv"] = p_inv
                    entry["model"] = model_name
                    entry["trait_type"] = trait_type
                    output_data.append(entry)

        return output_data
    except Exception as e:
        print(f"Error processing {person.get('id', '?')}: {e}")
        raise


def process_task(args):
    model, trait_type, person, questions, trait_data = args
    return process_single_person((person, questions, trait_data, trait_type, model))


def main():
    persona_data, trait_data = load_data()
    print(f"Loaded {len(persona_data)} personas")

    all_tasks = []
    for model in MODELS:
        for trait_type in TRAIT_TYPES:
            questions = load_items(trait_type, model)
            if not questions:
                print(f"  No items for {trait_type}/{model}, skipping")
                continue
            print(f"  {trait_type}/{model}: {sum(len(q['correlations']['positive']) + len(q['correlations']['negative']) for q in questions)} items")
            for person in persona_data:
                all_tasks.append((model, trait_type, person, questions, trait_data))

    print(f"Processing {len(all_tasks)} tasks...")
    num_processes = max(1, int(cpu_count() * 0.75))
    results = {}

    with Pool(processes=num_processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_task, all_tasks),
            total=len(all_tasks),
            desc="Overall Progress"
        ):
            for item in result:
                key = f"{item['trait_type']}_{item['model']}"
                results.setdefault(key, []).append(item)

    for key, data in results.items():
        output_path = Path(OUTPUT_DIR) / f"{key}_prompts.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} items to {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
