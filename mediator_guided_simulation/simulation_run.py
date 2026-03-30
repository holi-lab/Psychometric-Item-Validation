import json
import logging
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ===================== CONFIGURATION =====================
MODEL = "gpt-4.1-mini"
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "prompts" / "free" # free, caps, item, wvs, no_mediator, sampling
OUTPUT_DIR = BASE_DIR / "data" / "simulation" / "mini" / "free" # free, caps, item, wvs, no_mediator, sampling

# Mediator keys to simulate (must match prompt_generator's MEDIATORS keys)
# Both normal and inverted (_inv) versions are automatically processed
MEDIATOR_KEYS = ["free"] # free, caps, item, wvs, no_mediator, sampling

# Auto-derive prompt keys (normal + inverted for each mediator)
PROMPT_KEYS = []
for _key in MEDIATOR_KEYS:
    PROMPT_KEYS.extend([_key, f"{_key}_inv"])

# Survey-specific answer choices and score mappings (third-person, matching the paper)
# 5-point scale for Big5/VIA, 6-point scale for PVQ
SURVEY_CONFIG = {
    "big5": {
        "enum": [
            "Very Accurate",
            "Moderately Accurate",
            "Neither inaccurate nor accurate",
            "Moderately Inaccurate",
            "Very Inaccurate",
        ],
        "scores": {
            "Very Accurate": 5,
            "Moderately Accurate": 4,
            "Neither inaccurate nor accurate": 3,
            "Moderately Inaccurate": 2,
            "Very Inaccurate": 1,
        },
    },
    "pvq": {
        "enum": [
            "Very Much Like Them",
            "Like Them",
            "Somewhat Like Them",
            "A Little Like Them",
            "Not Like Them",
            "Not Like Them at All",
        ],
        "scores": {
            "Very Much Like Them": 6,
            "Like Them": 5,
            "Somewhat Like Them": 4,
            "A Little Like Them": 3,
            "Not Like Them": 2,
            "Not Like Them at All": 1,
        },
    },
    "via": {
        "enum": [
            "Very Much Like Them",
            "Like Them",
            "Neutral",
            "Unlike Them",
            "Very Much Unlike Them",
        ],
        "scores": {
            "Very Much Like Them": 5,
            "Like Them": 4,
            "Neutral": 3,
            "Unlike Them": 2,
            "Very Much Unlike Them": 1,
        },
    },
}
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpt_processing.log'),
        logging.StreamHandler(),
    ]
)

client = OpenAI()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def call_llm(prompt, trait_type):
    """Call LLM with structured output matching the paper's answer choices."""
    cfg = SURVEY_CONFIG[trait_type]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The selected answer.",
                            "enum": cfg["enum"],
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0,
    )

    ans_text = json.loads(response.choices[0].message.content)["answer"]
    return cfg["scores"][ans_text]


def process_single_item(item):
    """Process a single prompt item through the LLM."""
    try:
        for prompt_key in PROMPT_KEYS:
            if prompt_key in item and isinstance(item[prompt_key], str):
                try:
                    result = call_llm(item[prompt_key], item["trait_type"])
                    item[prompt_key] = result
                except Exception as e:
                    logging.error(f"Error {item['person_id']} {prompt_key}: {e}")
        return item
    except Exception as e:
        logging.error(f"Failed {item.get('person_id', '?')}: {e}")
        return None


def main():
    logging.info("Starting simulation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(INPUT_DIR.glob("*.json"))
    logging.info(f"Found {len(json_files)} files in {INPUT_DIR}")

    for file_path in json_files:
        logging.info(f"Processing {file_path.name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            all_items = json.load(f)

        logging.info(f"{len(all_items)} items in {file_path.name}")

        num_processes = max(1, int(cpu_count() * 0.75))
        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_item, all_items),
                total=len(all_items),
                desc=f"Processing {file_path.name}",
            ))

        processed = [r for r in results if r is not None]
        output_path = OUTPUT_DIR / file_path.name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        failed = len(results) - len(processed)
        logging.info(f"Done {file_path.name}: {len(processed)} ok, {failed} failed")


if __name__ == "__main__":
    main()
