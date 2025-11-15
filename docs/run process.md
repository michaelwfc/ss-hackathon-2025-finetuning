# Step 0: Initial Setup

check login in information from commpany

# Step 1: Build Dataset

check docs/build_dataset.md


# Step 2: Model Training using Amazon SageMaker 



## updata dataset version
s3://aileaguebucket-us-east-1-058264337588/dzd-4w5yohedtzc0dj/51wetw2mr1wmcn/dev/dataset/state_data_gen_cleaned2.jsonl

s3://aileaguebucket-us-east-1-058264337588/dzd-4w5yohedtzc0dj/51wetw2mr1wmcn/dev/dataset/synthtic_dataset_qwen_max_web_version_01.jsonl
s3://aileaguebucket-us-east-1-058264337588/dzd-4w5yohedtzc0dj/51wetw2mr1wmcn/dev/dataset/v2_qwen_max_web_and_sample.jsonl
s3://aileaguebucket-us-east-1-058264337588/dzd-4w5yohedtzc0dj/51wetw2mr1wmcn/dev/dataset/v3_synthetic_credit_risk_data_validated_shuffled.jsonl
s3://aileaguebucket-us-east-1-058264337588/dzd-4w5yohedtzc0dj/51wetw2mr1wmcn/dev/dataset/v4_0_credit_risk_data_validated_shuffled.jsonl
s3://aileaguebucket-us-east-1-058264337588/dzd-4w5yohedtzc0dj/51wetw2mr1wmcn/dev/dataset/v5_synthetic_data.jsonl
s3://aileaguebucket-us-east-1-058264337588/dzd-4w5yohedtzc0dj/51wetw2mr1wmcn/dev/dataset/v5_synthetic_data_validated_shuffled.jsonl

s3://aileaguebucket-us-east-1-058264337588/dzd-4w5yohedtzc0dj/51wetw2mr1wmcn/dev/output

## Set TrainingInstance to

Llama 3.2 3B instruct



ml.g5.2xlarge
(NOT the default option, which is ml.g5.12xlarge)


# Step3: Deploying Your Model

Model name: feicheng-model-02-eval

Instance Type: ml.g5.2xlarge

Endpoint name : select "Enter endpoint name"
Custom Endpoint name:  feicheng-model-02-eval


Endpoint name : jumpstart-dft-llama-3-2-3b-instruct-20251021-175054

SageMaker endpoint ARN :
arn:aws:sagemaker:us-east-1:058264337588:endpoint/jumpstart-dft-llama-3-2-3b-instruct-20251021-165015
arn:aws:sagemaker:us-east-1:058264337588:endpoint/jumpstart-dft-llama-3-2-3b-instruct-20251021-175054

arn:aws:sagemaker:us-east-1:058264337588:endpoint/feicheng-model-04-0
arn:aws:sagemaker:us-east-1:058264337588:endpoint/feicheng-model-00

# Step 5: Register and Submit Your Model

## Login to the leaderboard
username: state360 
password: <PASSWORD>



##  Artifact Management -> Register New Artifact 

## Leaderboard Submission


Enter event ID:
