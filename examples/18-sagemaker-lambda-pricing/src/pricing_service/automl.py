"""Train a demand model with SageMaker Autopilot (AutoML V2) and deploy the best candidate.

    python -m pricing_service.automl upload   # write synthetic training CSV to S3
    python -m pricing_service.automl start    # start the AutoML job
    python -m pricing_service.automl deploy <job-name>
Needs AWS credentials and the variables in .env.example.
"""
import io
import os
import sys
import time
from datetime import UTC, datetime

import boto3

from .features import TARGET
from .training_data import make_training_frame


def upload_training_data(s3, bucket: str, key: str = "demand/train/train.csv", df=None) -> str:
    df = make_training_frame() if df is None else df
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode(), ContentType="text/csv")
    return f"s3://{bucket}/{key.rsplit('/', 1)[0]}/"


def build_job_request(job_name: str, train_s3_prefix: str, output_s3_path: str, role_arn: str,
                      max_runtime_s: int = 3600, mode: str = "ENSEMBLING") -> dict:
    """Request for create_auto_ml_job_v2: tabular regression on units sold, optimised for MAE."""
    return {
        "AutoMLJobName": job_name,
        "AutoMLJobInputDataConfig": [{
            "ChannelType": "training",
            "ContentType": "text/csv;header=present",
            "CompressionType": "None",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": train_s3_prefix}},
        }],
        "OutputDataConfig": {"S3OutputPath": output_s3_path},
        "AutoMLProblemTypeConfig": {"TabularJobConfig": {
            "TargetAttributeName": TARGET,
            "ProblemType": "Regression",
            "Mode": mode,
            "CompletionCriteria": {"MaxAutoMLJobRuntimeInSeconds": max_runtime_s},
        }},
        "AutoMLJobObjective": {"MetricName": "MAE"},
        "DataSplitConfig": {"ValidationFraction": 0.2},
        "RoleArn": role_arn,
    }


def start_job(sm, train_s3_prefix: str, output_s3_path: str, role_arn: str, **kwargs) -> str:
    name = "demand-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    sm.create_auto_ml_job_v2(**build_job_request(name, train_s3_prefix, output_s3_path, role_arn, **kwargs))
    return name


def wait_for_job(sm, job_name: str, poll_s: int = 60, sleep=time.sleep) -> dict:
    while True:
        job = sm.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        status = job["AutoMLJobStatus"]
        if status == "Completed":
            return job
        if status in ("Failed", "Stopped"):
            raise RuntimeError(f"AutoML job {job_name} {status}: {job.get('FailureReason', '')}")
        sleep(poll_s)


def deploy_best_candidate(sm, job: dict, endpoint_name: str, role_arn: str,
                          instance_type: str = "ml.m5.large") -> str:
    """Create model, endpoint config and endpoint from the job's best candidate."""
    best = job["BestCandidate"]
    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    model_name = f"{endpoint_name}-{suffix}"
    sm.create_model(ModelName=model_name, ExecutionRoleArn=role_arn,
                    Containers=best["InferenceContainers"])
    sm.create_endpoint_config(EndpointConfigName=model_name, ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": model_name,
        "InstanceType": instance_type, "InitialInstanceCount": 1}])
    existing = sm.list_endpoints(NameContains=endpoint_name)["Endpoints"]
    if any(e["EndpointName"] == endpoint_name for e in existing):
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=model_name)
    else:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=model_name)
    return model_name


def main(argv: list[str]) -> None:  # pragma: no cover - needs AWS
    session = boto3.session.Session(region_name=os.environ.get("AWS_REGION"))
    sm, s3 = session.client("sagemaker"), session.client("s3")
    bucket, role = os.environ["TRAINING_BUCKET"], os.environ["SAGEMAKER_ROLE_ARN"]
    if argv[0] == "upload":
        print(upload_training_data(s3, bucket))
    elif argv[0] == "start":
        print(start_job(sm, f"s3://{bucket}/demand/train/", f"s3://{bucket}/demand/output/", role))
    elif argv[0] == "deploy":
        job = wait_for_job(sm, argv[1])
        print(deploy_best_candidate(sm, job, os.environ.get("SAGEMAKER_ENDPOINT_NAME", "demand-forecast"), role))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
