import os
import json
from openai import OpenAI
import random
from utils import extract_json_objects,add_json_objects, load_env, get_env, load_jsonl


load_env()

dashscope_api_key = get_env("DASHSCOPE_API_KEY")
model = get_env("DASHSCOPE_MODEL")
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


client = OpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



SYSTEM_PROMPT = """
You are a senior institutional credit risk analyst writing supervised fine-tuning data
to train a financial reasoning model.

You will be shown several sample JSON objects with this format:
{
  "instruction": "<analyst query>",
  "context": "",
  "response": "<expert-level credit analysis, 200–350 words>"
}

Your task is to generate *new examples* in the same style, tone, structure, and realism.

Guidelines:
- Use realistic credit risk language (CDS spreads, leverage, VaR, LTV, etc.)
- Include 3–5 quantitative values (basis points, %, $, ratios)
- Keep “context” empty
- Be specific and professional (no fluff, no generic advice)
- Output only valid JSON objects, one per line
"""


def generate_from_examples(samples, output_path, num_new=2,sample_size=10,):
    # Step 2: Select a few random reference examples
    reference_text = "\n".join(json.dumps(s, ensure_ascii=False) for s in random.sample(samples, 10))

    # Step 3: Build user message
    user_prompt = f"""
Below are {sample_size} reference examples of credit risk analysis data:

{reference_text}

Now generate {num_new} new examples that follow the same JSON structure and analytical depth.
Each must be a valid JSON object on its own line.
"""

    # Step 4: Call LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        top_p=0.9,
    )

    raw_ouput = response.choices[0].message.content.strip()
    # Step 5: Parse and save output
    examples = extract_json_objects(raw_ouput)
    saved_examples= add_json_objects(examples, output_path)
            
    print(f"✅ Generated {saved_examples.__len__()} new synthetic samples → {output_path}")


def build_generate_dataset(output_file="v5_synthetic_data.jsonl", output_dir="data",total_batches=2, num_new=2):
  
    output_path = os.path.join(output_dir, output_file)
    
    samples = load_jsonl("data/state_data_gen_cleaned2.jsonl")
    for i in range(total_batches):
        print(
        f"🧩 Generating batch {i+1}/{total_batches}"
    )
           
        generate_from_examples(samples=samples,  output_path=output_path, num_new=num_new,sample_size=10)


if __name__ == "__main__":
    output_file="v5_synthetic_data.jsonl"
    build_generate_dataset(total_batches=100, output_file=output_file, output_dir="data", num_new=2)

    
