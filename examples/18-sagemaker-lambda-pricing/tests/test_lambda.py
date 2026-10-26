import io
import json

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

from pricing_service import lambda_function as lf
from pricing_service.features import FEATURES, to_csv_rows

REQUEST = {"sku": "SKU-1", "store_id": "S01", "date": "2026-11-27", "current_price": 10.0,
           "unit_cost": 6.0, "inventory": 80, "competitor_price": 10.0, "is_holiday": False}


class FakeRuntime:
    """Stands in for the SageMaker endpoint: demand = 100 * (price / 10) ** -2."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def invoke_endpoint(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ClientError({"Error": {"Code": "ModelError", "Message": "boom"}}, "InvokeEndpoint")
        rows = [line.split(",") for line in kwargs["Body"].strip().splitlines()]
        out = "\n".join(str(100 * (float(r[FEATURES.index("price")]) / 10) ** -2) for r in rows).encode()
        return {"Body": StreamingBody(io.BytesIO(out), len(out)), "ContentType": "text/csv"}


@pytest.fixture
def clients(monkeypatch):
    runtime = FakeRuntime()
    cloudwatch = boto3.client("cloudwatch")
    stub = Stubber(cloudwatch)
    monkeypatch.setattr(lf, "_clients", {"sagemaker-runtime": runtime, "cloudwatch": cloudwatch})
    with stub:
        yield runtime, stub


def test_api_gateway_event_returns_profit_maximising_price(clients):
    runtime, stub = clients
    stub.add_response("put_metric_data", {}, {"Namespace": "Example/DynamicPricing", "MetricData": ANY})
    resp = lf.lambda_handler({"body": json.dumps(REQUEST)}, None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    # demand at 10.00 is 100 units but only 80 are in stock, so a higher price earns more
    assert body["recommended_price"] > 10.0
    assert body["recommended_price"] <= 11.5  # 15 % above competitor
    assert body["price_range"] == [8.0, 11.5]  # 20 % max change, 15 % above competitor
    call = runtime.calls[0]
    assert call["EndpointName"] == "demand-forecast" and call["ContentType"] == "text/csv"
    assert len(call["Body"].strip().splitlines()) == 21
    stub.assert_no_pending_responses()


def test_direct_invocation_and_holiday_flag_reach_the_model(clients):
    runtime, stub = clients
    stub.add_response("put_metric_data", {})
    lf.lambda_handler({**REQUEST, "is_holiday": True}, None)
    first_row = runtime.calls[0]["Body"].splitlines()[0].split(",")
    assert first_row[FEATURES.index("is_holiday")] == "1"
    assert first_row[FEATURES.index("day_of_week")] == "4"  # 2026-11-27 is a Friday


@pytest.mark.parametrize("bad", [{"current_price": "10"}, {"inventory": -1}, {"date": "27.11.2026"},
                                 {"sku": None}])
def test_invalid_requests_get_400(bad):
    resp = lf.lambda_handler({"body": json.dumps({**REQUEST, **bad})}, None)
    assert resp["statusCode"] == 400


def test_endpoint_error_gives_502(monkeypatch):
    monkeypatch.setattr(lf, "_clients", {"sagemaker-runtime": FakeRuntime(fail=True)})
    assert lf.lambda_handler(REQUEST, None)["statusCode"] == 502


def test_metric_failure_does_not_block_decision(clients):
    _, stub = clients
    stub.add_client_error("put_metric_data", "Throttling")
    assert lf.lambda_handler(REQUEST, None)["statusCode"] == 200


def test_demand_is_made_monotone_and_stock_caps_sales():
    req = lf.parse_event(dict(REQUEST))
    prices = lf.price_corridor(req)
    decision = lf.choose_price(req, prices, prices * 0 + 200)
    assert decision["stock_limited"] and decision["recommended_price"] == prices[-1]


def test_csv_rows_reject_injection():
    with pytest.raises(ValueError):
        to_csv_rows([["a,b", "S01", 1, 1, 0, 1.0, 1.0, 1.0]])
