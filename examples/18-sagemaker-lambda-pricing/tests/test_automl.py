import boto3
import pytest
from botocore.stub import ANY, Stubber

from pricing_service.automl import (
    build_job_request,
    deploy_best_candidate,
    start_job,
    upload_training_data,
    wait_for_job,
)
from pricing_service.features import FEATURES, TARGET
from pricing_service.training_data import make_training_frame

ROLE = "arn:aws:iam::123456789012:role/ExampleSageMakerRole"
ARN = "arn:aws:sagemaker:eu-central-1:123456789012:"


def test_training_frame_has_the_inference_column_order():
    df = make_training_frame(days=30)
    assert list(df.columns) == [*FEATURES, TARGET]
    assert len(df) == 30 * 4 and (df[TARGET] >= 0).all()


def test_upload_writes_csv_with_header():
    s3 = boto3.client("s3")
    with Stubber(s3) as stub:
        stub.add_response("put_object", {}, {"Bucket": "example-bucket", "Key": "demand/train/train.csv",
                                             "Body": ANY, "ContentType": "text/csv"})
        prefix = upload_training_data(s3, "example-bucket", df=make_training_frame(days=5))
    assert prefix == "s3://example-bucket/demand/train/"


def test_job_request_is_valid_for_the_sagemaker_api():
    # Stubber validates the parameters against the real service model of botocore
    sm = boto3.client("sagemaker")
    with Stubber(sm) as stub:
        stub.add_response("create_auto_ml_job_v2", {"AutoMLJobArn": ARN + "automl-job/x"})
        name = start_job(sm, "s3://example-bucket/demand/train/", "s3://example-bucket/out/", ROLE)
    assert name.startswith("demand-")
    req = build_job_request("j", "s3://b/p/", "s3://b/o/", ROLE)
    cfg = req["AutoMLProblemTypeConfig"]["TabularJobConfig"]
    assert cfg["ProblemType"] == "Regression" and cfg["TargetAttributeName"] == TARGET


def test_wait_for_job_raises_on_failure():
    sm = boto3.client("sagemaker")
    with Stubber(sm) as stub:
        channel = {"ChannelType": "training", "ContentType": "text/csv;header=present",
                   "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://b/p/"}}}
        base = {"AutoMLJobName": "j", "AutoMLJobArn": ARN + "automl-job/j", "AutoMLJobInputDataConfig": [channel],
                "OutputDataConfig": {"S3OutputPath": "s3://b/o/"}, "RoleArn": ROLE,
                "CreationTime": "2026-01-01", "LastModifiedTime": "2026-01-01",
                "AutoMLJobSecondaryStatus": "Starting"}
        stub.add_response("describe_auto_ml_job_v2", {**base, "AutoMLJobStatus": "InProgress"})
        stub.add_response("describe_auto_ml_job_v2",
                          {**base, "AutoMLJobStatus": "Failed", "FailureReason": "bad data"})
        with pytest.raises(RuntimeError, match="bad data"):
            wait_for_job(sm, "j", sleep=lambda s: None)


def test_deploy_creates_model_config_and_endpoint():
    sm = boto3.client("sagemaker")
    containers = [{"Image": "123.dkr.ecr.eu-central-1.amazonaws.com/autogluon:latest",
                   "ModelDataUrl": "s3://b/o/model.tar.gz", "Environment": {}}]
    job = {"BestCandidate": {"CandidateName": "best", "InferenceContainers": containers}}
    with Stubber(sm) as stub:
        stub.add_response("create_model", {"ModelArn": ARN + "model/m"},
                          {"ModelName": ANY, "ExecutionRoleArn": ROLE, "Containers": containers})
        stub.add_response("create_endpoint_config", {"EndpointConfigArn": ARN + "endpoint-config/c"})
        stub.add_response("list_endpoints", {"Endpoints": []})
        stub.add_response("create_endpoint", {"EndpointArn": ARN + "endpoint/e"},
                          {"EndpointName": "demand-forecast", "EndpointConfigName": ANY})
        name = deploy_best_candidate(sm, job, "demand-forecast", ROLE)
        stub.assert_no_pending_responses()
    assert name.startswith("demand-forecast-")
