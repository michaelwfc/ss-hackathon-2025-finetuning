import os, json, random, time
import re
from openai import OpenAI
from utils import load_env, get_env, extract_json_objects,add_json_objects

load_env()

dashscope_api_key = get_env("DASHSCOPE_API_KEY")
model = get_env("DASHSCOPE_MODEL")
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


client = OpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# Fixed system message – defines global tone and rules
SYSTEM_MESSAGE = """
You are a **senior institutional credit risk analyst** at a global asset manager.
You write internal research notes and portfolio risk assessments.
Your tone is **analytical, professional, and concise**, as if preparing a memo for a Chief Risk Officer.

Your task: generate synthetic supervised fine-tuning data to train a compact model
to emulate expert financial reasoning in **credit and counterparty risk analysis**.

Every output must consist of **2 JSON objects**, each with:
- "instruction": a realistic analyst task or query.
- "context": always "".
- "response": a concise expert analysis (150–300 words),
  identifying risk factors, interpreting data, quantifying exposures,
  and concluding with specific recommendations or outcomes.

Each example should vary by portfolio type, market driver, and risk dimension.
Use realistic quantitative reasoning (basis points, percentages, durations).
Never add explanations or meta-comments outside the JSON objects.
"""



# Diversity control pools
SECTORS = [
    "investment-grade corporate bonds",
    "high-yield corporate debt",
    "leveraged loans",
    "bank loan portfolios",
    "emerging market sovereign bonds",
    "emerging market corporate debt",
    "municipal revenue bonds",
    "municipal general obligation bonds",
    "commercial real estate debt (CMBS)",
    "residential mortgage-backed securities (RMBS)",
    "collateralized loan obligations (CLO)",
    "asset-backed securities (ABS)",
    "structured credit products",
    "private credit / direct lending",
    "distressed debt",
    "convertible bonds",
    "subordinated bank debt (AT1/Tier 2)",
    "covered bonds",
    "securitized credit",
    "infrastructure project finance",
]

RISK_TYPES = [
    "credit spread volatility",
    "default probability escalation",
    "rating migration (downgrade risk)",
    "counterparty credit deterioration",
    "settlement and operational risk",
    "liquidity and redemption pressure",
    "refinancing and rollover risk",
    "covenant breach exposure",
    "cross-default contagion",
    "wrong-way risk in derivatives",
    "concentration risk (single-name or sector)",
    "correlation breakdown during stress",
    "recovery rate uncertainty",
    "sovereign-corporate linkage risk",
    "currency-induced credit stress",
    "collateral quality deterioration",
    "mark-to-market volatility",
    "funding gap and maturity mismatch",
]

MARKET_DRIVERS = [
    "Federal Reserve aggressive tightening cycle (300+ bps)",
    "yield curve inversion signaling recession",
    "credit spread widening across high-yield (200+ bps)",
    "energy sector stress from oil price collapse",
    "commercial real estate valuation decline (20-30%)",
    "regional banking crisis and deposit flight",
    "emerging market currency depreciation wave",
    "sovereign debt crisis in major economy",
    "corporate earnings recession with margin compression",
    "leveraged buyout financing drought",
    "CLO repricing due to rising default rates",
    "municipal fiscal stress from tax revenue decline",
    "geopolitical conflict disrupting supply chains",
    "inflation persistence requiring extended high rates",
    "quantitative tightening reducing market liquidity",
    "credit rating agency methodology changes",
    "ESG-driven financing constraint for fossil fuel issuers",
    "bank capital requirements tightening (Basel IV implementation)",
    "prime brokerage consolidation after counterparty failure",
    "fallen angel wave from BBB downgrades to high-yield",
]

MARKET_DRIVERS = [
    "yield curve inversion",
    "downgrade wave",
    "oil price collapse",
    "policy tightening cycle",
    "sovereign default risk",
    "spread compression",
    "geopolitical conflict",
    "rating agency downgrades",
    "USD appreciation",
    "energy price shock",
    "funding stress in repo markets",
    "widening HY spreads",
]


REGIONS = [
    "United States",
    "Western Europe (Eurozone)",
    "United Kingdom",
    "Emerging Asia (ex-Japan)",
    "Latin America",
    "Middle East and North Africa",
    "Sub-Saharan Africa",
    "Central and Eastern Europe",
    "Developed Asia-Pacific (Australia, Japan, Singapore)",
    "Global diversified portfolio",
]

PORTFOLIO_CONTEXTS = [
    "a $2.8B institutional fixed income portfolio",
    "a $450M pension fund credit allocation",
    "a $1.2B insurance company investment portfolio",
    "a $650M multi-strategy hedge fund credit book",
    "a $5.5B asset manager's corporate bond fund",
    "a $320M family office alternative credit portfolio",
    "a $180M endowment's fixed income holdings",
    "a $900M CLO warehouse facility",
    "a $2.1B bank's loan portfolio",
    "a $750M emerging market debt fund",
]

ANALYTICAL_DIMENSIONS = [
    "duration and spread sensitivity",
    "covenant structures and compliance headroom",
    "collateral quality and loan-to-value ratios",
    "debt service coverage and interest coverage metrics",
    "liquidity ratios and cash conversion cycles",
    "leverage multiples (debt/EBITDA) and capital structure",
    "recovery rate assumptions under stress scenarios",
    "credit migration probability matrices",
    "correlation and portfolio concentration metrics",
    "refinancing schedules and maturity walls",
]


def create_user_prompt(
    sector, risk_type, market_driver, region, portfolio_context, analytical_dim
):
    """Generate highly specific, context-rich user prompts"""

    return f"""Generate 2 synthetic training examples for institutional credit risk analysis with the following specifications:

**PORTFOLIO CONTEXT**: {portfolio_context}
**PRIMARY SECTOR FOCUS**: {sector}
**DOMINANT RISK TYPE**: {risk_type}
**MARKET ENVIRONMENT**: {market_driver}
**GEOGRAPHIC EXPOSURE**: {region}
**ANALYTICAL DIMENSION**: Focus particularly on {analytical_dim}

# SCENARIO REQUIREMENTS

Example 1: **Portfolio-Level Strategic Assessment**
- Frame as a quarterly risk review or portfolio rebalancing decision
- Include multi-factor risk assessment across 3-4 dimensions
- Quantify portfolio-wide impact (VaR, expected loss, correlation effects)
- Provide strategic recommendations with 3-6 month implementation horizon
- Reference specific position sizes (absolute $ or % of portfolio)

Example 2: **Position-Specific Tactical Analysis**
- Frame as an individual credit or counterparty assessment
- Deep dive into 2-3 security-specific risk factors
- Include credit metrics: spreads, ratings, leverage ratios, coverage ratios
- Provide tactical trading recommendations (buy/sell/hedge decisions)
- Specify monitoring triggers and exit criteria

# REALISM GUIDELINES

- Use realistic portfolio sizes: institutional = $500M-$5B, specialized = $100M-$800M
- Credit spreads: IG = 50-200bps, HY = 300-800bps, Distressed = 1000+ bps
- Typical leverage: IG corporate 2-3x, HY 4-6x, Stressed 6x+
- Duration ranges: Short 1-3yr, Intermediate 3-7yr, Long 7-15yr
- Reference real market instruments: CDX indices, CDS, specific bond structures

# PROHIBITED PATTERNS (Avoid These)

❌ Generic advice: "Monitor the situation closely"
❌ Vague sizing: "Reduce exposure somewhat"  
❌ Missing numbers: "Spreads have widened significantly"
❌ No timeline: "Implement hedging strategy"
❌ Qualitative only: "Risk is elevated" (without quantification)

✅ Instead: "Reduce energy HY exposure from 18% to 12% ($65M to $42M) over 60 days. Implement CDS protection on $20M of BB-rated midstream names. Monitor weekly for 50bps+ spread widening as exit trigger."

Generate examples that a real credit analyst would write before a risk committee meeting."""





def generate_data(output_file="v3_synthetic_credit_risk_data.jsonl", total_batches=5):
    output_path = os.path.join(OUTPUT_DIR, output_file)

    for i in range(total_batches):  # roughly 100 batches × 100 samples = 10k
        sector = random.choice(SECTORS)
        risk_type = random.choice(RISK_TYPES)
        market_driver = random.choice(MARKET_DRIVERS)
        region = random.choice(REGIONS)

        print(
            f"🧩 Generating batch {i+1}/{total_batches} [{sector}, {risk_type}, {market_driver}, {region}]"
        )
        try:

            user_message = f"""
              Generate 100 synthetic supervised training examples for credit risk analysis.
              Focus this batch on {sector} portfolios with primary exposure to {risk_type} risk,
              under a scenario characterized by {market_driver} in the {region} region.
              Ensure quantitative realism and portfolio management perspective.
              """

            response = client.chat.completions.create(
                model=model,  # "qwen3-max",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE.strip()},
                    {"role": "user", "content": user_message.strip()},
                ],
                temperature=1.0,
                top_p=0.9,
            )

            raw_output = response.choices[0].message.content.strip()
            

            # Append valid examples to the .jsonl file
            examples = extract_json_objects(raw_output)
            add_json_objects(examples, output_path)
        except Exception as e:
            print(f"⚠️ Error in batch {i+1}: {e}")

            time.sleep(random.uniform(2, 5))  # polite rate limiting


def build_batch_generate_job(
    output_file="batch_generate_job.jsonl", total_jobs=50, output_dir="data"
):
    """
    Build a batch generation job file in the required format for Qwen API batch processing.

    Args:
        output_file (str): Name of the output file
        total_jobs (int): Number of batch jobs to generate
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Generate batch jobs
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(total_jobs):
            sector = random.choice(SECTORS)
            risk_type = random.choice(RISK_TYPES)
            market_driver = random.choice(MARKET_DRIVERS)
            region = random.choice(REGIONS)

            # Create the user message
            user_message = f"""
              Generate 100 synthetic supervised training examples for credit risk analysis.
              Focus this batch on {sector} portfolios with primary exposure to {risk_type} risk,
              under a scenario characterized by {market_driver} in the {region} region.
              Ensure quantitative realism and portfolio management perspective.
              """

            # Create the job object
            job = {
                "custom_id": str(i + 1),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE.strip()},
                        {"role": "user", "content": user_message.strip()},
                    ],
                    "temperature": 1.0,
                    "top_p": 0.9,
                },
            }

            # Write job to file as JSONL
            f.write(json.dumps(job, ensure_ascii=True) + "\n")

    print(f"✅ Generated {total_jobs} batch jobs and saved to {output_path}")


if __name__ == "__main__":
    # generate_data(output_file="v3_1_synthetic_credit_risk_data.jsonl",total_batches=50)
    build_batch_generate_job(
        output_file="batch_generate_job.jsonl", output_dir="tasks", total_jobs=5
    )
