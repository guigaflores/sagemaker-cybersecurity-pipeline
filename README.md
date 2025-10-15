# Cybersecurity Threat Detection Pipeline on AWS
Building and deploying a Cybersecurity Threat Detection System using Amazon SageMaker.

## Project Overview
This project involves building and deploying a Cybersecurity Threat Detection System using Amazon SageMaker. The system identifies anomalous network activity that may indicate cyberattacks, such as DDoS attacks, unauthorized access, or phishing attempts. The machine learning pipeline automates data ingestion, preprocessing, model training, deployment, and inference.

### Key components include:
**Data Ingestion & Preprocessing**: Raw network traffic logs are collected, transformed, and feature-engineered to create a structured dataset.  
**Model Training & Evaluation**: An XGBoost model is trained to classify network activity as normal or malicious.  
**Deployment & Inference**: The trained model is deployed as an endpoint to detect real-time security threats.  
**Pipeline Automation**: An end-to-end SageMaker Pipeline automates data transformation, model training, and deployment.  

## Architecture
The following diagram illustrates the AWS architecture used in this project:
<img width="1052" height="486" alt="project-ai-cybersecurity-diagram" src="https://github.com/user-attachments/assets/3b776fca-de97-4e0c-a1b7-f7442d0e36d5" />

### Services Used

**Amazon SageMaker**: Trains, deploys, and serves the machine learning model. [Machine Learning]  
**Amazon S3**: Stores raw network traffic logs, preprocessed data, and model artifacts. [Storage]  
**AWS Lambda**: Automates data preprocessing tasks and feature extraction. [Compute]  
**Amazon CloudWatch**: Monitors model performance and logs security threats. [Monitoring]  
**AWS IAM**: Manages permissions and security policies for accessing AWS services. [Security]  

### How It Works

1. Raw network logs are ingested into Amazon S3.
2. Lambda processes and transforms logs into a structured dataset.
3. The processed dataset is stored back in a new S3 bucket.
4. SageMaker Pipelines train the XGBoost model.
5. Trained model artifacts are saved to S3 for deployment.
6. The model is deployed as a SageMaker Endpoint to detect security threats.  

*IAM manages access permissions, while CloudWatch collects and monitors system metrics.

## Project Setup
1. Preprocess Data and Feature Engineering
2. Training and Testing a Model using XGBoost
3. Deploy and Serve the Model
4. Automating with SageMaker Pipelines

## 1. Preprocess Data and Feature Engineering
Prepare network traffic data to train a machine learning model for cybersecurity threat detection. Set up the AWS environment, fetch a public dataset, clean the data, engineer meaningful features, and save the processed data for training.  
This is a critical step - high-quality input leads to a reliable and accurate model.

### 1.1 Create an IAM Role for SageMaker
SageMaker needs permissions to access other AWS services (like S3). Select AmazonSageMakerFullAccess and AmazonS3FullAccess permission policies.

### 1.2 Set Up Amazon SageMaker Notebook Instance
A Notebook Instance is an easy way to run Jupyter notebooks in AWS. Create a Notebook instance and select the SageMaker role created before.

### 1.3 Download & Upload a Public Dataset to S3
To test this project we'll use a public dataset from UNSW-NB15, which contains normal and malicious network activities. You can download the file UNSWNB15_training-set.csv in this repo.  
Create a S3 bucket and upload the file in a folder called ‘raw-data’ (for example).

### 1.4 Load & Explore the Dataset in SageMaker
Open a new notebook (conda_python3) inside the notebook instance created before and rename it ‘data_preprocessing.ipynb’.  
After that, paste the below code and run it:  

```python
import boto3
import pandas as pd
import io

# Setup S3 client
s3_client = boto3.client('s3')

# Download the file into memory
response = s3_client.get_object(Bucket='cybersecurity-sagemaker-ml-data1', Key='raw-data/UNSW_NB15_training-set.csv')

# Read it into pandas
df = pd.read_csv(io.BytesIO(response['Body'].read()))

# Explore
print(df.shape)
print(df.columns)
print(df.head())
print(df['label'].value_counts())
```
Explanation: we use S3 to store data and access it from the notebook.  
**label** column represents wheter traffic is normal (0) or malicious (1)  

### 1.5 Clean, Feature Engineer, Encode, and Normalize Data
Prepare the dataset so that the ML models can use it effectively.  

**Drop irrelevant columns**
- **id** is just a row number and has no predictive value.
- **attack_cat** is a more detailed attack type label, but for binary classification (label = malicious or normal), it’s not needed.

**Feature engineering (before encoding & scaling)**  
Create new features from existing ones to help the model detect patterns:
- **byte_ratio** – Source bytes / (Destination bytes + 1)  
  Helps detect traffic that’s heavily one-sided.+1 prevents division by zero.
- **is_common_port** – 1 if destination port is 80, 443, or 22  
  Flags traffic over common HTTP/HTTPS/SSH ports.
- **flow_intensity** – (Source packets + Destination packets) / (Duration + 1e-6)  
  Measures packet rate; useful for spotting floods or spikes.Small number (1e-6) prevents division by zero.

**One-hot encode categorical features**  
- Converts text columns like proto, service, and state into multiple binary columns.  
- Each unique category becomes its own column with 1 (present) or 0 (absent).

**Convert booleans to integers**  
- Some one-hot columns might be True/False.
- We explicitly convert them to 1/0 for compatibility with ML models.

**Scale numerical features**  
- Standardize all numeric columns except label so they have mean 0 and standard deviation 1.  
- Prevents features with larger ranges (e.g., byte counts) from dominating smaller-range features (e.g., ratios).

**Sanity checks**  
- Print final dataset shape, column names, a preview of the first rows, and class distribution in label.

```python
# --- 1. Drop irrelevant columns ---
df = df.drop(columns=['id', 'attack_cat'])

# --- 2. Feature engineering BEFORE encoding/scaling ---
df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
df['is_common_port'] = df['ct_dst_sport_ltm'].isin([80, 443, 22]).astype(int)
df['flow_intensity'] = (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-6)

# --- 3. One-hot encode categorical columns ---
categorical_cols = ['proto', 'service', 'state']
df = pd.get_dummies(df, columns=categorical_cols)

# Convert booleans to ints
df = df.astype({col: 'int' for col in df.columns if df[col].dtype == 'bool'})

# --- 4. Scale numerical features (except label) ---
from sklearn.preprocessing import StandardScaler

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('label')

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# --- Checks ---
print(df.shape)                                  # Final number of rows & columns
print(df.head())                                 # Preview first few rows
print(df.describe().T[['mean', 'std']])          # Confirm scaling stats
print(df[numerical_cols].mean().round(3))        # Should be ~0
print(df[numerical_cols].std().round(3))         # Should be ~1
```
**Conceptual Notes**  
- **One-hot encoding**: This takes a column like "proto with values TCP, UDP, and ICMP", and turns it into 3 new columns - each showing 1 or 0.  
- **StandardScaler**: It transforms values so they have a mean of 0 and standard deviation of 1 useful when features have wildly different units (like packet count vs. byte size).

**Tips for Feature Engineering**  
- Always divide by +1 or add a tiny number to avoid division by zero.
- Try plotting distributions of these new features — sometimes it helps visualize how well they separate classes.

### 1.6 Save the Preprocessed Data to S3  
Save your processed dataset locally as CSV.  
Upload it to your S3 bucket using SageMaker’s session utility.  
```python
import sagemaker
from sagemaker import get_execution_role

# Create SageMaker session and define bucket
session = sagemaker.Session()
bucket = 'cybersecurity-sagemaker-ml-data1'  # Replace with your actual S3 bucket name
processed_prefix = 'processed-data'      # Folder in S3 to store processed files

# Save preprocessed data locally
df.to_csv('preprocessed_data.csv', index=False)

# Upload to S3 inside the 'processed-data/' folder
s3_path = session.upload_data(
    path='preprocessed_data.csv',
    bucket=bucket,
    key_prefix=processed_prefix
)

print(f"Preprocessed data uploaded to: {s3_path}")
```
*Validate your S3 Bucket if the ‘processed-data.csv' file got uploaded under the ‘processed-data’ folder.  

## 2. Training and Testing a Model using XGBoost  
Train a machine learning model (using XGBoost) to classify whether a given network activity is normal or malicious, based on the features extracted in Step 1.  
We’ll use Amazon SageMaker’s built-in XGBoost algorithm, which makes training efficient and scalable.  

### 2.1 Load Preprocessed Data from S3  
Paste this code in a new cell and run:
```python
import pandas as pd
import boto3
import sagemaker

# Set up session and bucket
session = sagemaker.Session()
bucket = 'cybersecurity-sagemaker-ml-data1'
processed_prefix = 'processed-data'

# Download preprocessed data from S3
s3 = boto3.client('s3')
file_name = 'preprocessed_data.csv'
s3.download_file(bucket, f'{processed_prefix}/{file_name}', file_name)

# Load into pandas
df = pd.read_csv(file_name)
df.head()
```
This code downloads a preprocessed CSV file from an S3 bucket and loads it into a Pandas DataFrame for inspection.  

<img width="1103" height="244" alt="2 1 Load Preprocessed Data from S3" src="https://github.com/user-attachments/assets/27193992-3e16-4c8d-8525-d044c43bfd99" />


### 2.2 Split Data into Train/Test Sets  
Paste this code in a new cell and run:
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import pandas as pd

# Load data
df = pd.read_csv('preprocessed_data.csv')
X = df.drop(columns=['label'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CSV for inspection
train_df = pd.concat([y_train, X_train], axis=1)
test_df = pd.concat([y_test, X_test], axis=1)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# LIBSVM for SageMaker - fixed version
dump_svmlight_file(X_train, y_train.values.ravel(), 'train.libsvm')
dump_svmlight_file(X_test, y_test.values.ravel(), 'test.libsvm')
```
This code loads a preprocessed CSV dataset, splits it into training and testing sets, saves them as CSV for inspection, and converts them into LIBSVM format for use with Amazon SageMaker.  

### 2.3 Upload Training and Test Data to S3  
Paste this code in a new cell and run:  
```python
import sagemaker

session = sagemaker.Session()
bucket = 'cybersecurity-sagemaker-ml-data1'
train_prefix = 'xgboost-data/train'
test_prefix = 'xgboost-data/test'

train_input = session.upload_data('train.libsvm', bucket=bucket, key_prefix=train_prefix)
test_input = session.upload_data('test.libsvm', bucket=bucket, key_prefix=test_prefix)

print(f"Training data: {train_input}")
print(f"Testing data: {test_input}")
```
This code uploads the LIBSVM-formatted training and testing data files to specified paths in an S3 bucket for use with Amazon SageMaker.  

### 2.4 Set Up the XGBoost Training Job  
Paste this code in a new cell and run:  
```python
from sagemaker import image_uris
from sagemaker.estimator import Estimator

xgboost_image_uri = image_uris.retrieve("xgboost", region=session.boto_region_name, version="1.3-1")

xgb = Estimator(
    image_uri=xgboost_image_uri,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://cybersecurity-sagemaker-ml-data1/xgboost-model-output',
    sagemaker_session=session
)

xgb.set_hyperparameters(
    objective='binary:logistic',
    num_round=100,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    verbosity=1
)
```
This code configures an XGBoost estimator in SageMaker by retrieving the appropriate container image, setting training resources and output path, and specifying hyperparameters for a binary classification task.  

### 2.5 Train the Model  
Paste this code in a new cell and run:  
```python
# Train using data channels
xgb.fit({'train': train_input, 'validation': test_input})
```
This code starts training the XGBoost model on SageMaker using the uploaded training and validation data from S3.  
**Note:** This step runs the training job on SageMaker’s managed infrastructure using the built-in XGBoost container.  

### 2.6 Evaluate Model Performance  
Once training is complete, we’ll download the trained model, make predictions on test data, and calculate accuracy.  
Firstly we need to install **xgboost** Python package.
```python
!pip install xgboost
```
Paste this code in a new cell and run:
```python
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load and convert data
train_data = pd.read_csv('train.csv', header=None, dtype=str)
test_data = pd.read_csv('test.csv', header=None, dtype=str)

# Convert all columns to numeric
train_data = train_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaNs
train_data = train_data.dropna()
test_data = test_data.dropna()

# Split into features (X) and labels (y)
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Convert to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Set parameters and train the model
params = {
    "objective": "binary:logistic",
    "max_depth": 5,
    "eta": 0.2,
    "gamma": 4,
    "min_child_weight": 6,
    "subsample": 0.8,
    "verbosity": 1
}

model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100)

# Predict
y_pred_prob = model.predict(dtest)
y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
This code trains an XGBoost model locally using the same parameters as the SageMaker model and evaluates its performance on the test set by computing accuracy and a classification report.  

<img width="436" height="176" alt="2 6 Evaluate Model Performance" src="https://github.com/user-attachments/assets/c5b165af-e70d-4c99-aa7b-266da28e91dc" />

## 3. Deploy and Serve the Model  
Deploy the trained model on Amazon SageMaker and expose it as an API endpoint for real-time cybersecurity threat detection.  

### 3.1 Create a SageMaker Model from the Trained Model Artifact  
Before deployment, register the trained model in SageMaker through the code below:
```python
import boto3
from sagemaker import image_uris

sagemaker_client = boto3.client("sagemaker")
region = "us-east-2"
bucket_name = "cybersecurity-sagemaker-ml-data1"
model_artifact = f"s3://cybersecurity-sagemaker-ml-data1/xgboost-model-output/sagemaker-xgboost-2025-10-13-13-41-53-148/output/model.tar.gz"
model_name = "cybersecurity-threat-xgboost"

# Get XGBoost image URI
image_uri = image_uris.retrieve("xgboost", region=region, version="1.3-1")

# Use actual IAM Role ARN
execution_role = "arn:aws:iam::008971670328:role/SageMakerCybersecurityRole"

# Register the model
response = sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": image_uri,
        "ModelDataUrl": model_artifact
    },
    ExecutionRoleArn=execution_role
)

print(f"Model {model_name} registered successfully in SageMaker!")
```
**Note**: Replace the following placeholders with your own configuration: 
- model_artifact = `<bucket-name>/<training-job-name>`  
- execution_role = `<sagemakercybersecurityrole-arn>`

### 3.2 Deploy the Model as a SageMaker Endpoint  
Create a real-time endpoint that applications can call for cybersecurity threat detection.  
```python
# Define model name if not already defined
model_name = "cybersecurity-threat-xgboost"

# Define endpoint configuration
endpoint_config_name = "cybersecurity-threat-config"

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "DefaultVariant",
            "ModelName": model_name,
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1
        }
    ]
)

# Deploy endpoint
endpoint_name = "cybersecurity-threat-endpoint"

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"Endpoint '{endpoint_name}' is being deployed. This may take a few minutes...")
```
To check the status of the created endpoint:
```python
import boto3

# Endpoint name (must be the same used in create_endpoint)
endpoint_name = "cybersecurity-threat-endpoint"

# Create the SageMaker client
sagemaker_client = boto3.client("sagemaker")

# Retrieve the current endpoint status
response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

# Extract the status
status = response["EndpointStatus"]

print(f"Current status of endpoint '{endpoint_name}': {status}")
```
Possible values:  
- Creating: SageMaker is still deploying the endpoint  
- InService: The endpoint is ready and available for inference  
- Updating: The endpoint is being updated  
- OutOfService: The endpoint is deactivated  
- Failed: An error occurred during the deployment

### 3.3 Test the Deployed Endpoint  
Once the endpoint is active, we send a sample request to check if it correctly detects threats.
```python
import boto3
import numpy as np

runtime_client = boto3.client("sagemaker-runtime")

# Sample input in CSV format
sample_input = "0.5,0.3,0.8,0.2,0.1,0.6,0.9,0.4"

# Invoke the endpoint
response = runtime_client.invoke_endpoint(
    EndpointName="cybersecurity-threat-endpoint",  # or use endpoint_name if defined
    ContentType="text/csv",
    Body=sample_input
)

# Get prediction from response
result = response["Body"].read().decode("utf-8")
prediction_score = float(result.strip())

# Interpret prediction
predicted_label = "THREAT" if prediction_score > 0.5 else "SAFE"

print(f"Prediction: {predicted_label}")
```
Output:  

<img width="1104" height="460" alt="3 3 Test the Deployed Endpoint" src="https://github.com/user-attachments/assets/a77f9e91-e0ac-4cbd-85a5-c695510ac5ed" />

## 4. Automating with SageMaker Pipelines  
This step connects all the earlier parts of our project into one automated, production-grade ML workflow. It automates the entire machine learning workflow, including data preprocessing, training, evaluation, and deployment using Amazon SageMaker Pipelines.  
It transitions our project from:  
Manual development/testing → Automated pipeline orchestration using Amazon SageMaker Pipelines, Lambda, and EventBridge.  

**Amazon SageMaker Pipelines** is a workflow automation tool that helps:  
- Streamline data preprocessing, model training, and deployment.  
- Maintain version control for models.  
- Automate model retraining when new data arrives.

### 4.1 Define the SageMaker Pipeline Workflow  
**Data Processing Step**: Load & preprocess raw data (S3 storage).  
**Training Step**: Train XGBoost model on preprocessed data.  
**Evaluation Step**: Check model performance (Accuracy, F1-score).  
**Conditional Deployment Step**: Deploy the model only if it meets accuracy requirements.  

### 4.2 Create a SageMaker Pipeline Definition  
Define a pipeline that includes data preprocessing, training, evaluation, and deployment.  
Paste the below code in the notebook for Automated SageMaker pipeline:  
```python
# SageMaker Pipeline
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris

# Setup
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = 'cybersecurity-sagemaker-ml-data1'

# Parameters
training_instance_type = ParameterString(
    name="TrainingInstanceType", 
    default_value="ml.m5.large"
)

# Use built-in XGBoost (same as your step 3.3)
xgb_estimator = sagemaker.estimator.Estimator(
    image_uri=image_uris.retrieve("xgboost", session.boto_region_name, version="1.3-1"),
    role=role,
    instance_count=1,
    instance_type=training_instance_type,
    output_path=f's3://{bucket}/pipeline-model-output/',
    hyperparameters={
        'objective': 'binary:logistic',
        'num_round': 100,
        'max_depth': 5,
        'eta': 0.2,
        'gamma': 4,
        'min_child_weight': 6,
        'subsample': 0.8,
        'verbosity': 1
    }
)

# Training Step
step_train = TrainingStep(
    name="TrainCybersecurityModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=f's3://{bucket}/xgboost-data/train/train.libsvm',
            content_type="text/libsvm"
        ),
        "validation": TrainingInput(
            s3_data=f's3://{bucket}/xgboost-data/test/test.libsvm',
            content_type="text/libsvm"
        )
    }
)

# Create Pipeline
pipeline = Pipeline(
    name="simple-cybersecurity-pipeline",
    parameters=[training_instance_type],
    steps=[step_train],
    sagemaker_session=session,
)

# Run Pipeline
def run_pipeline():
    pipeline.upsert(role_arn=role)
    print("Pipeline created successfully!")
    
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    return execution

print("Automated pipeline ready! Run: execution = run_pipeline()")
```
This pipeline automates the machine learning workflow for cybersecurity threat detection by creating a reusable, one-click training process.  

What the Pipeline Does:  
- **Automated Training:** Automatically trains the XGBoost model using your preprocessed cybersecurity data from S3, eliminating the need to manually run training jobs.
- **Consistent Results:** Ensures the same hyperparameters and data sources are used every time, providing reproducible model training.
- **Parameterized Workflow:** Allows easy modification of training instance types and other parameters without changing the core pipeline code.
- **Scalable Process:** Can be easily re-run when new threat data becomes available, enabling continuous model updates as cybersecurity patterns evolve.

### 4.3 Trigger the Pipeline Execution  
Once the pipeline is defined, trigger execution to run the entire workflow in the next cell:  
```python
execution = run_pipeline()
print("SageMaker Pipeline Execution Started!")
```
Check status in another cell:
```python
status = execution.describe()['PipelineExecutionStatus']
print(f"Pipeline Status: {status}")
```

<img width="1150" height="395" alt="4 3 Trigger the Pipeline Execution" src="https://github.com/user-attachments/assets/b20c91cc-1571-499a-96d1-e07c41ebcb36" />

### 4.4 Automate Retraining with AWS EventBridge  

**Create a Lambda Function**  
Runtime = Python 3.9  
Select "Create a new role with basic Lambda permissions"  
Replace the default code with the code below (lambda_function.py):  
```python
import json
import boto3

def lambda_handler(event, context):
# Initialize SageMaker client
    sagemaker_client = boto3.client("sagemaker")

    try:
# Start your pipeline
        response = sagemaker_client.start_pipeline_execution(
            PipelineName="simple-cybersecurity-pipeline"# Your pipeline name
        )

        print(f"Pipeline started: {response['PipelineExecutionArn']}")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Pipeline execution started successfully",
                "executionArn": response['PipelineExecutionArn']
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error starting pipeline: {str(e)}")
        }
```

**Add SageMaker permissions to Lambda**  
In the “Configuration” tab of the Lambda function, click on the execution role name and attach the “AmazonSageMakerFullAccess” policy.  

**Create EventBridge Rule**  
Rule type = Rule with an event pattern → Custom pattern  
Event pattern:  
```json
{
  "source": ["aws.s3"],
  "detail-type": ["Object Created"],
  "detail": {
    "bucket": {
      "name": ["cybersecurity-sagemaker-ml-data1"]
    },
    "object": {
      "key": [{
        "prefix": "new-data/"
      }]
    }
  }
}
```
In the Target, select the Lambda function created before.  

### 4.5 Test the Automation  
Enable S3 Event Notifications (S3 Bucket → Properties → Event Notifications)  

**Create a new event notification**:  
Name: new-data-notification  
Prefix: new-data/  
Event types: Check "All object create events"  
Destination: "Lambda"  

**Test the Setup**  
Create a test file in your notebook:
```python
# Test automation
import boto3

s3_client = boto3.client('s3')

# Upload a test file to trigger the pipeline
test_content = "This is test data for pipeline automation"
s3_client.put_object(
    Bucket='cybersecurity-sagemaker-ml-data1',
    Key='new-data/test-trigger.txt',
    Body=test_content
)

print("Test file uploaded! Check Lambda logs to see if pipeline triggered.")
```

What this automation does:
1. New Data Arrives → File uploaded to s3://cybersecurity-sagemaker-ml-data1/new-data/
2. S3 Notifies EventBridge → S3 sends event to EventBridge
3. EventBridge Triggers Lambda → Lambda function receives the event
4. Lambda Starts Pipeline → SageMaker pipeline begins training
5. Automatic Retraining → Model updates without manual intervation

<img width="1903" height="679" alt="4 5 Test the Automation 2" src="https://github.com/user-attachments/assets/ffae8ed8-9b00-4938-be1c-3515ba5a6c09" />

<img width="1141" height="577" alt="4 5 Test the Automation 1" src="https://github.com/user-attachments/assets/896e75c1-cca8-4166-be96-4af369a3ca66" />

## 5. Conclusion
This project demonstrates how to build and deploy **a machine learning-based cybersecurity threat detection system** using Amazon SageMaker and other AWS services. By integrating **Amazon S3, Amazon SageMaker, AWS Lambda, Amazon CloudWatch, and SageMaker Pipelines**, we created a robust, scalable, and automated ML pipeline capable of detecting malicious network activity in near real-time.

**Key outcomes include**:
- **Data preprocessing and feature engineering** using Pandas and SageMaker processing jobs.
- **Training and evaluation of an XGBoost model** to classify network traffic as benign or malicious.
- **Deployment of a real-time inference endpoint** using SageMaker for live threat detection.
- **Automation of the ML workflow via SageMaker Pipelines** for consistent and repeatable model building.
- **End-to-end integration of cloud-native services** with secure, serverless infrastructure.

## Learnings
This hands-on project gave me valuable experience in building production-grade ML solutions in the cloud, particularly focused on cybersecurity threat detection and response. It also laid the foundation for integrating advanced features like real-time alerting, continuous training, and threat visualization dashboards.
