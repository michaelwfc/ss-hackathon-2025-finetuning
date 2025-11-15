import json
import random
import time
import os
from openai import OpenAI
from utils import get_env, load_env, require_env

load_env()

# Initialize client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

dashscope_api_key = get_env("DASHSCOPE_API_KEY")
model = get_env("DASHSCOPE_MODEL")
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# ENHANCED DIVERSITY CONTROL POOLS
# ============================================================================

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

# ============================================================================
# ENHANCED SYSTEM PROMPT - More Specific, Structured Guidance
# ============================================================================


def build_system_message(num_examples: int):
    return f"""
You are a **Principal Credit Risk Analyst** at a top-tier global institutional asset manager with over 20 years of experience in credit markets, counterparty risk, and portfolio construction.

Your task is to generate **high-quality supervised fine-tuning data** for a model specializing in institutional credit risk analysis. Each output must mirror the rigor of real risk committee papers, CRO memos, and portfolio briefings.

---

### OUTPUT REQUIREMENTS

Generate exactly **{num_examples} JSON objects**, each with this structure:

{{
  "instruction": "<realistic analyst query or task>",
  "context": "",
  "response": "<expert-level analysis, 200–350 words>"
}}

---

### RESPONSE FRAMEWORK

1. **Risk Identification & Quantification (30–40%)**
   - Identify 2–4 major credit risk factors.
   - Use quantitative data (basis points, percentages, ratios, etc.).
   - Reference realistic market thresholds.

2. **Causal Analysis & Market Context (20–30%)**
   - Explain transmission mechanisms of the risk.
   - Link macro factors to portfolio-level outcomes.
   - Discuss secondary effects or correlations.

3. **Actionable Recommendations (30–40%)**
   - Recommend portfolio adjustments (sizing, hedging, timeline).
   - Include numeric details: $ values, % allocations, bp changes.
   - Specify monitoring or escalation triggers.

4. **Quantified Outcomes (10–20%)**
   - Quantify expected P&L, VaR, or exposure impact.
   - Include stress-test scenarios and probability estimates.

---

### QUALITY STANDARDS

- **Realistic**: Use actual credit market terms (CDS spreads, OAS, LTV, CET1, etc.).
- **Specific**: Avoid vague language — always include numbers and timeframes.
- **Actionable**: Each recommendation must be implementable.
- **Professional Tone**: Objective, structured, and analytical.
- **Quantitative Depth**: Include 3–5 metrics per response.

---

### VARIABILITY REQUIREMENTS
Vary across examples:
- Portfolio size ($100M–$5B+)
- Credit quality (AAA–distressed)
- Time horizon (short-term to 18 months)
- Analytical lens (bottom-up vs. top-down)
- Context (routine monitoring vs. stress response)

Output **only valid JSON objects**, no markdown or explanatory text.
"""


# ============================================================================
# ENHANCED USER PROMPT TEMPLATE - Context-Rich Scenario Building
# ============================================================================


def create_user_prompt(
    sector,
    risk_type,
    market_driver,
    region,
    portfolio_context,
    analytical_dim,
    num_examples=2,  # New parameter with default
):
    """
    Generate context-rich prompts for synthetic institutional credit risk data generation.

    Parameters
    ----------
    sector : str
        Primary sector of focus (e.g., Energy, Technology).
    risk_type : str
        Dominant credit risk type (e.g., Default Risk, Liquidity Risk).
    market_driver : str
        Key macro or market driver (e.g., Rate Hikes, Commodity Prices).
    region : str
        Geographic exposure or focus (e.g., North America, APAC).
    portfolio_context : str
        Description of the portfolio context (e.g., diversified fund, leveraged loan portfolio).
    analytical_dim : str
        Analytical dimension (e.g., correlation, exposure concentration, duration risk).
    num_examples : int, optional
        Number of realistic synthetic examples to generate (default: 2).
    """

    return f"""
Generate **{num_examples} realistic training examples** for institutional credit risk analysis based on the parameters below.

---

### CONTEXT PARAMETERS
- **Portfolio Context:** {portfolio_context}
- **Primary Sector:** {sector}
- **Dominant Risk Type:** {risk_type}
- **Market Environment:** {market_driver}
- **Geographic Exposure:** {region}
- **Analytical Focus:** {analytical_dim}

---

### TASK INSTRUCTIONS

For each example, produce a JSON object with:
{{
  "instruction": "<credit analyst question or task>",
  "context": "",
  "response": "<expert analysis, 200–350 words following the risk framework>"
}}


---

### REALISM REQUIREMENTS
- Institutional portfolio size: $500M–$5B; specialized credit: $100M–$800M.
- Typical spreads: IG = 50–200bps; HY = 300–800bps; Distressed = 1000+bps.
- Typical leverage: IG = 2–3×; HY = 4–6×; Stressed = 6×+.
- Durations: short = 1–3y, intermediate = 3–7y, long = 7–15y.
- Use real instruments (CDX, CDS, bonds, CLO tranches, etc.).

---


### STYLE AND QUALITY RULES
- Write in the tone of a professional institutional credit analyst.
- Always include 3–5 quantitative metrics.
- Avoid vague phrases (e.g., “reduce exposure somewhat”, “monitor closely”).
- Always specify **how much**, **how fast**, and **why**.
- Example of correct phrasing:
  “Reduce energy HY exposure from 18% to 12% ($65M → $42M) over 60 days.
   Hedge $20M BB-rated midstream names with 12M CDS at 400bps.
   Exit trigger: +50bps spread widening.”

---

Generate {num_examples} examples as if written for a risk committee meeting.
Output **only valid JSON objects** — no markdown, comments, or extra text.
"""


def extract_json_objects(text):
    """Enhanced JSON extraction with better error handling"""
    # Remove markdown code blocks if present
    text = text.replace("```json", "").replace("```", "")

    examples = []
    brace_count = 0
    current_obj = ""
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            current_obj += char
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            current_obj += char
            continue

        if char == '"' and not escape_next:
            in_string = not in_string

        if not in_string:
            if char == "{":
                if brace_count == 0:
                    current_obj = ""
                brace_count += 1
            elif char == "}":
                brace_count -= 1

        current_obj += char

        if brace_count == 0 and current_obj.strip():
            try:
                obj = json.loads(current_obj.strip())
                # Validate required fields
                if all(k in obj for k in ["instruction", "context", "response"]):
                    # Ensure context is empty string
                    obj["context"] = ""
                    # Validate response length (should be substantial)
                    if len(obj["response"]) > 100:
                        examples.append(obj)
            except json.JSONDecodeError:
                pass
            current_obj = ""

    return examples


# ============================================================================
# MAIN GENERATION FUNCTION - Enhanced with Better Sampling
# ============================================================================


def generate_data(output_file, num_examples, total_batches):
    """Generate dataset with improved diversity and quality control"""

    successful_batches = 0
    total_examples = 0
    
    system_message = build_system_message(num_examples)

    for i in range(total_batches):
        # Stratified sampling to ensure diversity
        sector = random.choice(SECTORS)
        risk_type = random.choice(RISK_TYPES)
        market_driver = random.choice(MARKET_DRIVERS)
        region = random.choice(REGIONS)
        portfolio_context = random.choice(PORTFOLIO_CONTEXTS)
        analytical_dim = random.choice(ANALYTICAL_DIMENSIONS)

        print(f"\n{'='*80}")
        print(f"🎯 Batch {i+1}/{total_batches}")
        print(f"   Sector: {sector}")
        print(f"   Risk: {risk_type}")
        print(f"   Driver: {market_driver}")
        print(f"   Region: {region}")
        print(f"{'='*80}")

        try:
            user_message = create_user_prompt(
                sector=sector,
                risk_type=risk_type,
                market_driver=market_driver,
                region=region,
                portfolio_context=portfolio_context,
                analytical_dim=analytical_dim,
                num_examples=num_examples,
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message.strip()},
                    {"role": "user", "content": user_message.strip()},
                ],
                temperature=0.9,  # Slightly lower for more consistent quality
                top_p=0.95,
            )

            raw_output = response.choices[0].message.content.strip()
            examples = extract_json_objects(raw_output)

            if examples:
                output_path = os.path.join(OUTPUT_DIR, output_file)
                with open(output_path, "a", encoding="utf-8") as f:
                    for ex in examples:
                        f.write(json.dumps(ex, ensure_ascii=True) + "\n")

                successful_batches += 1
                total_examples += len(examples)
                print(f"✅ Saved {len(examples)} examples | Total: {total_examples}")
            else:
                print(f"⚠️  No valid examples extracted")

        except Exception as e:
            print(f"❌ Error in batch {i+1}: {e}")

        # Polite rate limiting
        time.sleep(random.uniform(2, 4))

    print(f"\n{'='*80}")
    print(f"🏁 Generation Complete!")
    print(f"   Successful batches: {successful_batches}/{total_batches}")
    print(f"   Total examples: {total_examples}")
    print(f"   Output file: {os.path.join(OUTPUT_DIR, output_file)}")
    print(f"{'='*80}")


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    output_file = "v4_credit_risk_data.jsonl"
    generate_data(
        output_file=output_file,
        num_examples=50,
        total_batches=100,  # Adjust based on your needs
    )
