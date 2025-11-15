## The hachathon topic: Credit Risk Analysis

CHALLENGE

Global asset managers and custodial banks are under mounting pressure to assess portfolio credit exposure, liquidity risk, and counterparty stability in real time. Traditional credit models based on historical analysis tend to interpret complex, multi-asset portfolios or to contextualize market events, leading to delayed or overly simplistic insights. While giant market intelligence platforms or massive foundation models can address this partially, their high computational and operational costs make them impractical for broad deployment across advisory and risk functions.

SOLUTION

Develop a "Portfolio Credit Risk Intelligence Assistant"—a fine-tuned 3B model trained on expertly synthesized financial reasoning data. This model will emulate the judgment of seasoned institutional risk analysts, focusing on multi-factor exposure analysis and stress testing rather than broad generic scenarios. The fine-tuned assistant should be able to interpret complex market information and generate actionable recommendations—providing risk-adjusted insights faster and more accessibly than larger general-purpose models.

OUTCOME

The resulting assistant will enable operations and risk teams to perform granular credit assessments with agility. It will surface vulnerabilities, potential exposures, threats, and articulate scenario-based recommendations. This approach creates a cost-effective, explainable AI tool for institutional portfolio management confidence in stability through data-driven foresight.

KEY SERVICE(S)

Amazon SageMaker, SageMaker Unified Studio & Amazon S3




## validate your dataset

Visit the leaderboard website (
https://d202uv4374t48v.cloudfront.net/
)
Navigate to the "Validate Training Data" section
Paste or upload your JSONL file
Click “Validate JSONL” to check for errors
If no errors are found, proceed to upload to S3 and fine-tune (Visit point 5 -
Amazon S3
section from
Step1
)

## PartyRock

The PartyRock app is a shortcut to doing so. But creating the dataset is the fundemtal apsect of the challenge. I'm not sure if we give the participants the example training data we created. I'll ask Akhil
 
###  Demo promt from AWS with PartyRock

"""
1. Here is the use case scenario for the finetuned dataset:
Global asset managers and custodial banks are under mounting pressure to assess portfolio credit exposure, liquidity risk, and counterparty stability in real time. Traditional credit models and generic AI assistants fail to interpret complex, multi-asset portfolios or to contextualize market events, leading to delayed or overly simplistic insights. While giant market intelligence platforms or massive foundation models can address this partially, their high computational and operational costs make them impractical for broad deployment across advisory and risk functions.

Develop a “Portfolio Credit Risk Intelligence Assistant”—a fine-tuned 3B model trained on expertly synthesized financial reasoning data. This model will emulate the judgment of seasoned institutional risk analysts, focusing on multi-factor exposure analysis and stress testing rather than broad, generic summaries. The fine-tuned assistant should be able to interpret complex market scenarios and generate actionable recommendations—providing risk-adjusted insights faster and more accessibly than larger general-purpose models.

2. Here is a description of the functionality of the finetuned model we'll be building with the finetuned data. And this contains the lens at which the questions are based from. i.e. from the lens of a real customer financial scenario, including their data, portfolio, and customer profile
Develop a “Portfolio Credit Risk Intelligence Assistant”—a fine-tuned 3B model trained on expertly synthesized financial reasoning data. This model will emulate the judgment of seasoned institutional risk analysts, focusing on multi-factor exposure analysis and stress testing rather than broad, generic summaries. The fine-tuned assistant should be able to interpret complex market scenarios and generate actionable recommendations—providing risk-adjusted insights faster and more accessibly than larger general-purpose models.

3. Here is an example of the type of question I expect our finetuned model to get and how to answer the question. Use this as a frame of reference for generating more relevant questions and maintain a high quality of response resolution explanation

"""

summary

{"instruction": "What are the key challenges faced by global asset managers in real-time portfolio risk assessment?", "context": "", "response": "Global asset managers face challenges in simultaneously evaluating credit exposure, liquidity risk, and counterparty stability. Traditional models lack capability to analyze complex multi-asset portfolios effectively, while advanced platforms are cost-prohibitive for widespread implementation."}
{"instruction": "What are the limitations of traditional credit models in modern portfolio management?", "context": "", "response": "Traditional credit models struggle with real-time analysis, lack multi-asset portfolio interpretation capabilities, and fail to incorporate market context effectively. They provide delayed insights and oversimplified risk assessments for complex investment strategies."}


