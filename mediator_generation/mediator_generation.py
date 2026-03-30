"""
Mediator Generation for Psychometric Item Validation

Generates mediators — situational or psychological factors that influence how
a psychological trait translates into observable behavior — using four strategies:
  - free:  Free-form conflict and persona generation (two-step: conflicting characteristics → persona sentences)
  - caps:  CAPS-based generation (two-step: 5 cognitive-affective processing system categories, each with conflicting characteristics → persona sentences)
  - item:  Questionnaire-item-based generation (two-step: conflicting mediator per item → persona sentence)
  - wvs:   World Values Survey-based generation (two-step: WVS questions → persona sentences, then conflict filtering per trait)

Usage:
  1. Set CONFIG below.
  2. Run:  python mediator_generation.py
"""

import json
import os
from typing import Dict, List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from multiprocessing import Pool

# ===================== CONFIGURATION =====================
CONFIG = {
    "method": "free",                     # "free" | "caps" | "item" | "wvs"
    "model": "gpt-4.1",                   # OpenAI model name
    "trait_json": "../traits_selection/trait.json",    # path to trait definitions
    "item_jsons": [                        # generated item files (used by "item" method)
        "../item_generation/big5_combined.json",
        "../item_generation/pvq_combined.json",
        "../item_generation/via_part1_combined.json",
        "../item_generation/via_part2_combined.json",
    ],
    "wvs_json": "../data/wvs_questionnaire/wvs_questionnaire.json",  # WVS questionnaire (used by "wvs" method)
    "output_dir": "./mediators",      # output directory
    "temperature": 1,
    "max_tokens": 16000,
}
# =========================================================

# ==================== CAPS CATEGORIES ====================
#  CAPS categories (used by "caps" method)
CAPS_CATEGORIES = [
    "Situation Encodings",
    "Expectancies and Beliefs",
    "Affective Responses",
    "Goals and Values",
    "Competencies and Self-regulatory Plans",
]
# =========================================================

# ==================== JSON SCHEMA ====================
PERSONA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "persona_sentences",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentences": {
                    "type": "array",
                    "description": "A collection of sentences that describe a persona.",
                    "items": {
                        "type": "string",
                        "description": "A single sentence describing an aspect of the persona.",
                    },
                }
            },
            "required": ["sentences"],
            "additionalProperties": False,
        },
    },
}

MEDIATOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mediator_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mediator": {
                    "type": "string",
                    "description": "A single personal characteristic, background factor, or life circumstance.",
                },
            },
            "required": ["mediator"],
            "additionalProperties": False,
        },
    },
}

WVS_PERSONA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "wvs_persona",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": "A single first-person sentence describing a persona who agrees with the WVS question.",
                },
            },
            "required": ["persona"],
            "additionalProperties": False,
        },
    },
}

CONFLICT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "conflict_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "conflicts": {
                    "type": "boolean",
                    "description": "Whether the persona's values conflict with accurate measurement of the personality trait.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for the conflict determination.",
                },
            },
            "required": ["conflicts", "reasoning"],
            "additionalProperties": False,
        },
    },
}
# =========================================================


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def load_traits(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_llm(client: OpenAI, prompt: str, response_format=None, max_tokens: int = None) -> str:
    params = dict(
        model=CONFIG["model"],
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=CONFIG["temperature"],
        max_tokens=max_tokens or CONFIG["max_tokens"],
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    if response_format is not None:
        params["response_format"] = response_format
    else:
        params["response_format"] = {"type": "text"}
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content


def parse_persona_json(raw: str) -> List[Dict]:
    try:
        data = json.loads(raw)
        return [{"persona": s} for s in data["sentences"]]
    except (json.JSONDecodeError, KeyError):
        print(f"Error parsing JSON response: {raw[:200]}")
        return []


def load_generated_items(item_jsons: List[str]) -> Dict[str, List[Dict]]:
    """Load generated items (excluding original 'psy' items) grouped by trait."""
    items_by_trait: Dict[str, List[Dict]] = {}
    for path in item_jsons:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            if item.get("source") == "psy":
                continue
            if item.get("expected_trait", "").lower() == "bogus":
                continue
            trait = item["expected_trait"]
            items_by_trait.setdefault(trait, []).append(item)
    return items_by_trait


def load_wvs(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pregenerate_wvs_personas(wvs_data: Dict, client: OpenAI) -> List[Dict]:
    """Step 1: Convert each WVS question into a first-person persona sentence."""
    personas = []
    total = len(wvs_data)
    for i, (q_id, q_data) in enumerate(wvs_data.items(), 1):
        topic = q_data["topic"]
        question = q_data["question"]
        print(f"  [WVS Step 1] ({i}/{total}) Generating persona for {q_id}: {topic}")
        prompt = (
            f"Create a concise persona sentence who say 'Yes/Agree/Likely' to the question below. "
            f'Use "I" as the subject.\n\n'
            f"Question Topic: {topic}\n"
            f"Question: {question}"
        )
        raw = call_llm(client, prompt, response_format=WVS_PERSONA_SCHEMA, max_tokens=256)
        try:
            data = json.loads(raw)
            personas.append({
                "q_id": q_id,
                "topic": topic,
                "question": question,
                "persona": data["persona"],
            })
        except (json.JSONDecodeError, KeyError):
            print(f"  Error parsing persona for WVS {q_id}: {raw[:200]}")
    print(f"  [WVS Step 1] Done: {len(personas)} persona sentences generated.")
    return personas


# ──────────────────────────────────────────────
#  Method: FREE
# ──────────────────────────────────────────────
def generate_free(trait_name: str, trait_def: str, client: OpenAI) -> List[Dict]:
    conflict_prompt = (
        f"Trait: {trait_name}\n"
        f"Definition: {trait_def}\n\n"
        f"List possible human characteristics or backgrounds that would be unlikely "
        f"or contradictory for someone who strongly values {trait_name}."
        f"Just number them without detailed explanation. Make many values as possible."
    )
    conflicts = call_llm(client, conflict_prompt, max_tokens=4096)

    persona_prompt = (
        f"For each of the contents below, write a single sentence introducing a "
        f'person\'s persona. Use "I" as a subject.\n\n'
        f'{conflicts}'
    )
    raw = call_llm(client, persona_prompt, response_format=PERSONA_SCHEMA, max_tokens=4096)
    return parse_persona_json(raw)


# ──────────────────────────────────────────────
#  Method: CAPS
# ──────────────────────────────────────────────
def generate_caps(trait_name: str, trait_def: str, client: OpenAI) -> List[Dict]:
    all_personas = []
    for category in CAPS_CATEGORIES:
        print(f"  [{trait_name}] Processing CAPS category: {category}")
        conflict_prompt = (
            f"Trait: {trait_name}\n"
            f"Definition: {trait_def}\n\n"
            f"List possible {category} that could create internal conflict or lead to "
            f"changes in behavior for someone who strongly values {trait_name}. "
            f"List as many items as possible. Number each item without detailed explanation."
        )
        conflicts = call_llm(client, conflict_prompt)

        persona_prompt = (
            f"For each of the contents below, write a single sentence introducing a "
            f'person\'s persona. Use "I" as a subject.\n\n'
            f'{conflicts}'
        )
        raw = call_llm(client, persona_prompt, response_format=PERSONA_SCHEMA)
        try:
            data = json.loads(raw)
            for s in data["sentences"]:
                all_personas.append({"category": category, "persona": s})
        except (json.JSONDecodeError, KeyError):
            print(f"  Error parsing JSON for category {category}")
    return all_personas


# ──────────────────────────────────────────────
#  Method: ITEM
# ──────────────────────────────────────────────
def generate_item(trait_name: str, trait_def: str, client: OpenAI,
                  items: List[Dict] = None) -> List[Dict]:
    if not items:
        print(f"  [{trait_name}] No generated items found, skipping.")
        return []

    results = []
    for item_entry in items:
        item_text = item_entry["item"]
        prompt = (
            f"Trait: {trait_name}\n"
            f"Definition: {trait_def}\n"
            f"Survey Item: {item_text}\n"
            f"Generate a single personal characteristic, background factor, or life circumstance "
            f"that could plausibly lead someone—even among people who highly value {trait_name}—"
            f"to respond contrary to the survey item above."
        )
        raw = call_llm(client, prompt, response_format=MEDIATOR_SCHEMA, max_tokens=512)
        try:
            data = json.loads(raw)
            mediator = data["mediator"]

            persona_prompt = (
                f"For each of the contents below, write a single sentence introducing a "
                f'person\'s persona. Use "I" as a subject.\n\n'
                f'{mediator}'
            )
            persona_raw = call_llm(client, persona_prompt, response_format=PERSONA_SCHEMA, max_tokens=512)
            try:
                persona_data = json.loads(persona_raw)
                persona = persona_data["persona"]
            except (json.JSONDecodeError, KeyError):
                print(f"  Error parsing persona for item '{item_text[:50]}': {persona_raw[:200]}")
                persona = ""

            results.append({
                "item": item_text,
                "persona": persona,
                "correlation_type": item_entry.get("expected_correlation"),
            })
        except (json.JSONDecodeError, KeyError):
            print(f"  Error parsing mediator for item '{item_text[:50]}': {raw[:200]}")
    return results


# ──────────────────────────────────────────────
#  Method: WVS
# ──────────────────────────────────────────────
def generate_wvs(trait_name: str, trait_def: str, client: OpenAI,
                 items: List[Dict] = None) -> List[Dict]:
    """Step 2: Filter pre-generated WVS personas by whether they conflict with the trait."""
    if not items:
        print(f"  [{trait_name}] No WVS personas found, skipping.")
        return []

    results = []
    for wvs_entry in items:
        persona = wvs_entry["persona"]
        prompt = (
            f"Consider the given values and the personality trait below.\n"
            f"Determine whether the given values conflict with the personality trait, "
            f"making it difficult for individuals to respond accurately to questions "
            f"designed to measure the trait.\n\n"
            f"<Personality Trait>\n"
            f"{trait_name}: {trait_def}\n\n"
            f"<Values>\n"
            f"{persona}"
        )
        raw = call_llm(client, prompt, response_format=CONFLICT_SCHEMA, max_tokens=512)
        try:
            data = json.loads(raw)
            if data["conflicts"]:
                results.append({
                    "question_id": wvs_entry["q_id"],
                    "persona": persona,
                    "reasoning": data["reasoning"],
                })
        except (json.JSONDecodeError, KeyError):
            print(f"  Error parsing conflict for WVS {wvs_entry['q_id']}: {raw[:200]}")
    return results


# ──────────────────────────────────────────────
#  Dispatcher
# ──────────────────────────────────────────────
GENERATORS = {
    "free": generate_free,
    "caps": generate_caps,
    "item": generate_item,
    "wvs": generate_wvs,
}


# ──────────────────────────────────────────────
#  Parallel processing
# ──────────────────────────────────────────────
def process_single_trait(args: Tuple) -> Tuple[str, List[Dict]]:
    index, total, trait_name, trait_def, method, items = args
    client = OpenAI()
    print(f"[{index}/{total}] Processing trait: {trait_name}  (method={method})")
    generator = GENERATORS[method]
    if method in ("item", "wvs"):
        personas = generator(trait_name, trait_def, client, items=items)
    else:
        personas = generator(trait_name, trait_def, client)
    print(f"[{index}/{total}] {trait_name}: generated {len(personas)} mediators")
    return (trait_name, personas)


def run(config: Dict) -> None:
    load_dotenv()
    method = config["method"]
    assert method in GENERATORS, f"Unknown method: {method}. Choose from {list(GENERATORS)}"

    traits_data = load_traits(config["trait_json"])

    all_traits = []
    for category_data in traits_data.values():
        for name, definition in category_data.items():
            all_traits.append((name, definition))

    # Pre-load data for "item" and "wvs" methods
    items_by_trait = {}
    if method == "item":
        items_by_trait = load_generated_items(config["item_jsons"])
        print(f"Loaded generated items for {len(items_by_trait)} traits")
    elif method == "wvs":
        client = OpenAI()
        wvs_data = load_wvs(config["wvs_json"])
        print(f"Loaded {len(wvs_data)} WVS questions. Running Step 1 (persona generation)...")
        wvs_personas = pregenerate_wvs_personas(wvs_data, client)
        items_by_trait = {name: wvs_personas for name, _ in all_traits}

    total = len(all_traits)
    print(f"Method: {method} | Traits: {total} | Model: {config['model']}")

    task_args = [
        (i + 1, total, name, defn, method, items_by_trait.get(name, []))
        for i, (name, defn) in enumerate(all_traits)
    ]

    with Pool() as pool:
        results = pool.map(process_single_trait, task_args)

    result = dict(results)

    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], f"mediator_{method}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(result)} traits → {output_path}")


if __name__ == "__main__":
    run(CONFIG)
