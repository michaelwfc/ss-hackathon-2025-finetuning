


# Sample data generate prompt

META_PROMPT_FOR_DATA_GENERATE = """
Now we are holding a hackathon and we need to generate some sample data to supervise finetuning a llama 3.2 3 B model to analyze credit risk 

# Hachathon topic : Credit Risk Analysis

- CHALLENGE

Global asset managers and custodial banks are under mounting pressure to assess portfolio credit exposure, liquidity risk, and counterparty stability in real time. Traditional credit models based on historical analysis tend to interpret complex, multi-asset portfolios or to contextualize market events, leading to delayed or overly simplistic insights. While giant market intelligence platforms or massive foundation models can address this partially, their high computational and operational costs make them impractical for broad deployment across advisory and risk functions.

- SOLUTION

Develop a "Portfolio Credit Risk Intelligence Assistant"—a fine-tuned 3B model trained on expertly synthesized financial reasoning data. This model will emulate the judgment of seasoned institutional risk analysts, focusing on multi-factor exposure analysis and stress testing rather than broad generic scenarios. The fine-tuned assistant should be able to interpret complex market information and generate actionable recommendations—providing risk-adjusted insights faster and more accessibly than larger general-purpose models.

- OUTCOME

The resulting assistant will enable operations and risk teams to perform granular credit assessments with agility. It will surface vulnerabilities, potential exposures, threats, and articulate scenario-based recommendations. This approach creates a cost-effective, explainable AI tool for institutional portfolio management confidence in stability through data-driven foresight.

# Output 
the sample data has 3 keys : instruction, context, response
each context is empty

the sample data is as follows:

{"instruction": "Analyze the credit risk implications of a $2.8B corporate bond portfolio with 35% exposure to BBB-rated energy sector bonds, given recent oil price volatility and rising interest rates.", "context": "", "response": "This portfolio faces elevated credit risk from concentrated BBB energy exposure during commodity stress. Key concerns: (1) BBB energy bonds approaching fallen angel threshold with oil weakness - estimate 12-18% of energy holdings at risk of downgrade to high yield within 6 months; (2) Duration risk amplifies losses with rising rates, particularly for lower-rated credits; (3) Liquidity deterioration likely in stressed energy names. Recommend: Reduce energy BBB exposure to <20% through selective sales of weakest credits, hedge duration risk on remaining energy positions, establish contingent liquidity facilities. Stress scenario suggests potential 8-12% portfolio drawdown if energy downgrades materialize concurrent with further rate increases."}
{"instruction": "Assess counterparty risk for our prime brokerage arrangements with three major investment banks, considering recent market volatility and their Q3 earnings reports.", "context": "", "response": "Counterparty risk profile shows divergent trajectories requiring exposure rebalancing. Bank A demonstrates solid capital but declining trading profitability signals business model stress - maintain current exposure but monitor quarterly. Bank B exhibits strongest risk-adjusted performance with stable trading revenues - consider increasing allocation within concentration limits. Bank C presents elevated concern with declining CET1 approaching regulatory comfort zone and rising credit provisions indicating asset quality deterioration - recommend reducing exposure by 30-40% over 60 days. Immediate actions: Implement daily mark-to-market monitoring for Bank C positions, negotiate additional collateral posting thresholds, and establish backup prime brokerage capacity. Overall counterparty concentration remains manageable but requires active rebalancing toward stronger credits."}
{"instruction": "Evaluate liquidity risk in our municipal bond portfolio during a potential recession scenario where tax revenues decline 20% across major holdings.", "context": "", "response": "Municipal liquidity risk is significantly elevated under this stress scenario, with fundamental and technical pressures converging. Primary concerns: (1) Illinois GOs face acute stress given existing fiscal challenges - 20% revenue decline could trigger rating actions and severe price dislocations; (2) MTA revenue bonds vulnerable to ridership-dependent cash flows during recession; (3) Technical liquidity deteriorating as fund outflows force selling in less liquid names. Credit differentiation: California/Texas holdings better positioned due to stronger fiscal reserves. Liquidity assessment: Expect 15-25% price volatility in weaker credits, bid-ask spreads widening 3-5x normal levels. Recommend: Pre-position to sell Illinois exposure while markets remain functional, increase cash allocation to 15% for opportunistic purchases, focus liquidity in highest-grade GO bonds from fiscally sound states. Consider reducing average maturity to improve liquidity profile during stress period."}
{"instruction": "Analyze credit exposure risk in our CLO equity tranches given rising default expectations and tightening lending standards in the leveraged loan market.", "context": "", "response": "CLO equity positions face substantial credit stress as leveraged loan fundamentals deteriorate rapidly. Risk analysis reveals: (1) Equity tranches will absorb first losses as defaults accelerate - current 4-5% default projection could eliminate 40-60% of distributions and trigger principal erosion in weaker vintage CLOs; (2) Refinancing risk increasing as 30% of underlying loans mature within 24 months amid tightening credit markets; (3) Manager quality divergence becoming critical - identify managers with superior workout capabilities. Vintage analysis: 2019-2020 vintages better positioned with higher loan prices and stronger coverage ratios, while 2021-2022 deals face compressed equity cushions. Immediate actions: Conduct manager-by-manager stress testing, prepare for significant distribution cuts, consider selective sales of weakest positions before broader distress materializes. Maintain positions only in top-quartile managers with proven special situations expertise. Expected loss scenario: 25-40% principal impairment across the portfolio over 18-month stress period."}
{"instruction": "What are the credit implications of our emerging market corporate bond exposure given potential US dollar strengthening and global recession concerns?", "context": "", "response": "EM corporate credit faces multifaceted stress from dollar strength and global recession, requiring urgent portfolio repositioning. Key vulnerabilities: (1) Local currency bonds face dual pressure from FX depreciation and credit deterioration - expect 20-30% volatility in total return terms; (2) Commodity-heavy exposure creates procyclical risk during global slowdown despite recent strength; (3) EM financial sector bonds vulnerable to domestic economic stress and potential capital flight. Regional analysis: Latin America faces acute currency pressure, Asia benefits from stronger fiscal positions, EMEA most exposed to European recession spillover. Credit migration risk significant with BB+/BB average rating - estimate 15-20% of holdings at risk of distressed levels. Strategic repositioning: Reduce local currency exposure to <25%, focus on USD-denominated investment grade names, trim commodity exposure to 20%, emphasize defensive sectors like utilities. Hedge residual FX exposure and establish higher cash buffers for volatility management. Timeline for rebalancing: 90 days before technical conditions deteriorate further."}


can you help generate a efficient and accurate  prompt for generating sample data first which will help  generate sample data to win the hachathon competition
    
"""




"""
**Optional refinement for competition-level results:**
Once you’ve generated 100–200 examples, filter or re-rank them with a second pass prompt such as:

> “Evaluate and keep only the top 25% of entries that show the highest financial reasoning quality, specificity, and actionable recommendations.”

This will yield a cleaner dataset for fine-tuning and demonstrate a reproducible pipeline to the judges — a big plus in hackathons.

---

Would you like me to extend this prompt into a *few-shot version* with 3–5 seed examples embedded to stabilize tone and reasoning style for generation? That’s usually what turns “good synthetic data” into “hackathon-winning data.”
"""



PROMPT_GENERATE_CREDIT_RISK_FINETUNING_DATA_v1 = """
You are a **senior institutional credit risk analyst** working for a global asset manager.
Your task is to **generate synthetic supervised fine-tuning data** to train a small model (3 B parameters) to emulate expert financial reasoning for **portfolio credit risk analysis**.
Generate outputs as if training data for an institutional-grade credit risk analysis model, prioritizing clarity, realism, and expert tone over verbosity.

Each output must be a **JSON object** with the following three fields:

* `"instruction"` — a realistic analyst task or query related to credit or counterparty risk, e.g., “Analyze credit risk implications of…” or “Evaluate liquidity risk under…”
* `"context"` — leave this as an empty string `""`.
* `"response"` — a concise, *expert-level* written analysis (150–300 words) that clearly identifies risk factors, interprets data, quantifies exposures where plausible, and ends with clear recommendations or scenario conclusions.

**Tone and content requirements:**

* Sound like a **seasoned portfolio risk professional**, not a textbook.
* Use **quantitative reasoning** (percentages, timeframes, magnitudes) and **domain language** (e.g., CET1, downgrade risk, duration risk, spread widening, liquidity stress).
* Each example must differ in **portfolio type**, **market driver**, and **risk dimension**. Vary by sector (corporate, sovereign, EM, CLO, muni, etc.), and by scenario (macro, rate, credit, liquidity, counterparty, geopolitical).
* Maintain realism: plausible market events, data, and managerial recommendations.
* Avoid boilerplate “depends on many factors” answers.

**Format:**
Generate between **100 JSON objects**, separated by newlines, for each call.
Each must follow this schema exactly:

```
{"instruction": "...", "context": "", "response": "..."}
```

**Example:**
{"instruction": "Assess the credit risk impact on a $1.2B sovereign bond portfolio following a 250bp rise in U.S. Treasury yields and widening EM spreads.", "context": "", "response": "The portfolio faces moderate mark-to-market losses from duration exposure and higher EM credit spread volatility... [continue with detailed expert analysis]"}
"""


PROMPT_GENERATE_CREDIT_RISK_FINETUNING_DATA_FEWSHOT_VERSION_V1 = """

You are a **senior institutional credit risk analyst** working for a global asset manager.
Your task is to **generate realistic supervised fine-tuning data** to train a 3B-parameter model for **portfolio credit risk analysis**.

Each output must be a **JSON object** with the following three fields:

* `"instruction"` — a realistic analyst task or query involving credit, counterparty, or liquidity risk.
* `"context"` — always an empty string `""`.
* `"response"` — a concise expert analysis (150–300 words) that diagnoses risk factors, quantifies exposure where plausible, and ends with actionable recommendations or scenario outcomes.

**Style guide:**

* Use the voice of a **seasoned institutional analyst**, not a textbook.
* Demonstrate reasoning under uncertainty and stress conditions.
* Include relevant quantitative markers: percentages, timeframes, or loss estimates.
* Vary by portfolio type (corporate, sovereign, EM, CLO, municipal, structured) and by stress driver (interest rates, spreads, FX, liquidity, downgrade cycles, geopolitical).
* Avoid generic phrasing or hedged “it depends” answers.
* Ensure all responses sound like internal portfolio risk memos or committee notes.

---

### **Seed Examples (Few-Shot Guidance)**

{"instruction": "Analyze the credit risk implications of a $2.8B corporate bond portfolio with 35% exposure to BBB-rated energy sector bonds, given recent oil price volatility and rising interest rates.", "context": "", "response": "This portfolio faces elevated credit risk from concentrated BBB energy exposure during commodity stress. Key concerns: (1) BBB energy bonds are near the fallen-angel threshold with oil weakness—roughly 15% of holdings risk downgrade within six months; (2) Rising yields worsen refinancing pressure for capital-intensive issuers; (3) Liquidity conditions in secondary markets deteriorate sharply in energy sell-offs. Recommend reducing energy BBB exposure from 35% to below 20%, hedge duration with pay-fixed swaps, and build a 10% liquidity reserve. Stress modeling indicates a potential 8–12% drawdown if downgrades coincide with further rate increases."}

{"instruction": "Assess counterparty risk for our prime brokerage relationships with three major investment banks following Q3 earnings reports and recent volatility in swap markets.", "context": "", "response": "Counterparty exposures diverge materially. Bank A’s capital ratios remain strong (CET1 14.6%) but its trading VaR doubled quarter-on-quarter—maintain exposure but tighten collateral triggers. Bank B exhibits consistent profitability and stable funding—consider raising exposure within limits. Bank C’s CET1 has slipped to 11.2% and loan-loss provisions jumped 30%, implying latent credit stress; reduce exposure 30% and request incremental margining. Across the portfolio, maintain daily mark-to-market surveillance and prepare backup prime lines to mitigate concentration risk."}

{"instruction": "Evaluate liquidity risk in our municipal bond holdings under a U.S. recession scenario with 20% tax revenue decline across key states.", "context": "", "response": "Liquidity stress in the muni book could intensify sharply. Illinois and New Jersey GOs would face immediate rating pressure given pre-existing fiscal weakness; spreads could widen 75–120 bps. Transit-linked revenue bonds may lose up to 25% of market value on ridership declines. California and Texas names remain more resilient due to stronger reserves. Anticipate bid-ask spreads widening 3–5x normal. Action plan: raise cash allocation to 15%, trim lower-rated GOs, shorten portfolio duration to increase flexibility, and maintain staggered exit points for weaker credits."}

{"instruction": "Analyze credit exposure in our CLO equity tranches amid tightening loan markets and rising default forecasts.", "context": "", "response": "CLO equity tranches are entering a high-stress phase as loan defaults rise from 2.5% to a projected 5–6%. Equity tranches absorb first losses, and cash-on-cash yields may fall 40–50% under a moderate stress path. Older 2019–2020 vintages with wider spreads and stronger collateral are more defensive, while 2021–2022 vintages carry thinner cushions and are more exposed. Recommend trimming exposure to underperforming managers, reinvesting only in top-quartile CLO platforms with proven restructuring capabilities. Expect 25–40% principal impairment across weakest vintages if default momentum continues."}

{"instruction": "Assess the credit implications of a $900M emerging-market sovereign bond portfolio facing U.S. dollar appreciation and capital outflows.", "context": "", "response": "The portfolio’s risk profile deteriorates under dollar strength. EM sovereigns with high external debt (Argentina, Egypt) face refinancing strain as yields widen 150–200 bps. Commodity exporters benefit partially from higher hard-currency revenues, but capital outflows may offset gains. Average duration of 6.2 years amplifies mark-to-market losses. Recommend shifting 20% into USD-denominated quasi-sovereigns with shorter duration, reducing local-currency exposure to below 25%, and deploying FX forwards to hedge currency losses. Under sustained dollar strength, anticipate 10–15% total return drawdown over the next 9–12 months."}

---

### **Your Task**

Using the style and structure shown above, **generate 5 new JSON objects** that fit the same schema but cover different portfolios, risk factors, and scenarios (e.g., corporate credit under geopolitical shock, bank funding stress, structured products during liquidity crunch, sovereign downgrade cycles, private credit deterioration, etc.).

Ensure outputs are **factually plausible**, **analytically rigorous**, and **consistent in tone** with the examples.
Output only newline-separated JSON objects—no commentary or explanations.





---

This version will reliably stabilize tone, structure, and vocabulary. The seed examples act as style anchors, so your model’s generations will stay disciplined and realistic while still creative across scenarios.

Would you like me to extend this with **sector balancing guidance** (so you can systematically cover 8–10 distinct market verticals in your dataset)? That helps when you scale beyond the first few hundred examples.

"""