# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

import litserve as ls
from litserve.specs.system_one import (
    DecisionRequest,
    SystemOneRequest,
    SystemOneSpec,
    _choice_confidence,
    _score_confidence,
    _to_decision_question,
)
from litserve.test_examples.system_one_spec_example import (
    TestDecisionAPI,
    TestDecisionAPIWithOutput,
    TestDecisionAPIWithUsage,
    TestDecisionAPIWithYieldEncodeResponse,
    TestDecisionAPIWithYieldPredict,
    TestDecisionAsyncAPI,
    TestDecisionBatchedAPI,
)
from litserve.utils import wrap_litserve_start

VALID_OUTPUT = {"department": [0, 0, 0], "escalate": [0, 0], "frustration": [0, 0, 0]}


def softmax(logits, temperature=1.0):
    exps = [math.exp(x / temperature) for x in logits]
    return [e / sum(exps) for e in exps]


@pytest.mark.asyncio
async def test_system_one_spec(system_one_request_data):
    server = ls.LitServer(TestDecisionAPI(spec=SystemOneSpec()))

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            resp = await ac.post("/v1/systemone", json=system_one_request_data, timeout=10)
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["model"] == "lit"
            assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}
            assert list(body["answers"]) == ["department", "escalate", "frustration"], "Answers keep request order"

            department = body["answers"]["department"]
            probs = softmax([0.0, -1.0, 2.0])
            assert department["type"] == "choice"
            assert department["choice"] == "billing"
            assert department["probabilities"] == pytest.approx(dict(zip(["returns", "shipping", "billing"], probs)))
            assert department["confidence"] == pytest.approx(_choice_confidence(probs))

            assert body["answers"]["escalate"] == {"type": "noul", "noul": pytest.approx(softmax([1.0, 0.0])[0])}

            frustration = body["answers"]["frustration"]
            probs = softmax([0.0, 1.0, 0.5])
            assert frustration["type"] == "score"
            assert frustration["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
            assert frustration["probabilities"] == pytest.approx(dict(zip(["0", "1", "2"], probs)))
            assert frustration["score"] == pytest.approx(probs[1] + 2 * probs[2])
            assert frustration["confidence"] == pytest.approx(_score_confidence(probs))


@pytest.mark.asyncio
async def test_system_one_spec_with_usage_and_temperature(system_one_request_data):
    server = ls.LitServer(TestDecisionAPIWithUsage(spec=SystemOneSpec(temperature=2.0)))

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            resp = await ac.post("/v1/systemone", json=system_one_request_data, timeout=10)
            assert resp.status_code == 200, resp.text
            assert resp.json()["usage"] == {"input_tokens": 42, "output_tokens": 3}
            assert resp.json()["answers"]["escalate"]["noul"] == pytest.approx(softmax([1.0, 0.0], 2.0)[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api",
    [
        TestDecisionBatchedAPI(spec=SystemOneSpec(), max_batch_size=2, batch_timeout=0.01),
        TestDecisionAsyncAPI(spec=SystemOneSpec(), enable_async=True),
    ],
)
async def test_system_one_spec_batched_and_async(api, system_one_request_data):
    server = ls.LitServer(api)

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            resp = await ac.post("/v1/systemone", json=system_one_request_data, timeout=10)
            assert resp.status_code == 200, resp.text
            assert resp.json()["answers"]["department"]["choice"] == "billing"


@pytest.mark.asyncio
async def test_system_one_spec_ignores_unknown_top_level_keys(system_one_request_data):
    server = ls.LitServer(TestDecisionAPI(spec=SystemOneSpec()))

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            payload = {**system_one_request_data, "session_id": "abc", "provider": {"order": ["lit"]}}
            resp = await ac.post("/v1/systemone", json=payload, timeout=10)
            assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("questions", "state"),
    [
        ({"q": {"type": "maybe", "instructions": "?"}}, "text"),
        ({"q": {"type": "noul"}}, "text"),
        ({"q": {"type": "noul", "instructions": "?", "options": ["yes", "no"]}}, "text"),
        ({"q": {"type": "choice", "criteria": {}}}, "text"),
        ({"q": {"type": "choice", "criteria": {"a": None, "b": None, "c": None, "d": None}}}, "text"),
        ({"q": {"type": "score", "criteria": ["only"]}}, "text"),
        ({"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}, "text"),
        ({f"q{i}": {"type": "noul", "instructions": "?"} for i in range(3)}, "text"),
        ({}, "text"),
        ({"q": {"type": "noul", "instructions": "?"}}, None),
    ],
)
async def test_system_one_spec_invalid_request(questions, state):
    # predict would fail the request with a 500, so a 422 also shows it never ran
    server = ls.LitServer(
        TestDecisionAPIWithOutput(output=None, spec=SystemOneSpec(max_questions=2, max_options=3)),
    )

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            payload = {"model": "lit", "state": state, "questions": questions}
            resp = await ac.post("/v1/systemone", json=payload, timeout=10)
            assert resp.status_code == 422, resp.text
            assert isinstance(resp.json()["detail"], list), "422 errors should list the invalid fields"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "error"),
    [
        ({"department": [0, 0, 0], "escalate": [0, 0]}, "frustration"),
        ({**VALID_OUTPUT, "extra": [0]}, "extra"),
        ({**VALID_OUTPUT, "escalate": [0, 0, 0]}, "escalate"),
        ({**VALID_OUTPUT, "department": {"returns": 0, "nope": 0, "billing": 0}}, "department"),
        ({**VALID_OUTPUT, "frustration": [0, float("nan"), 0]}, "frustration"),
    ],
)
async def test_system_one_spec_invalid_predict_output(output, error, system_one_request_data):
    server = ls.LitServer(TestDecisionAPIWithOutput(output=output, spec=SystemOneSpec()))

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            resp = await ac.post("/v1/systemone", json=system_one_request_data, timeout=10)
            assert resp.status_code == 500, resp.text
            assert error in resp.json()["detail"], "The error should name the question"


@pytest.mark.asyncio
async def test_system_one_spec_non_dict_output(system_one_request_data):
    server = ls.LitServer(TestDecisionAPIWithOutput(output=[1, 2, 3], spec=SystemOneSpec()))

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            resp = await ac.post("/v1/systemone", json=system_one_request_data, timeout=10)
            assert resp.status_code == 500, resp.text


@pytest.mark.asyncio
async def test_system_one_spec_custom_api_path(system_one_request_data):
    with pytest.warns(UserWarning, match="Custom API path detected"):
        server = ls.LitServer(TestDecisionAPI(spec=SystemOneSpec(), api_path="/v1/decisions"))

    with wrap_litserve_start(server) as server:
        async with (
            LifespanManager(server.app) as manager,
            AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac,
        ):
            resp = await ac.post("/v1/decisions", json=system_one_request_data, timeout=10)
            assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("api_class", [TestDecisionAPIWithYieldPredict, TestDecisionAPIWithYieldEncodeResponse])
def test_system_one_spec_rejects_streaming(api_class):
    with pytest.raises(ValueError, match="doesn't support streaming"):
        ls.LitServer(api_class(spec=SystemOneSpec()))


def test_system_one_spec_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="temperature must be greater than 0"):
        SystemOneSpec(temperature=0)


def test_system_one_decode_request(system_one_request_data):
    request = SystemOneSpec().decode_request(SystemOneRequest(**system_one_request_data))
    assert isinstance(request, DecisionRequest)
    department, escalate, frustration = request.questions
    assert department.options == ("returns", "shipping", "billing")
    assert department.descriptions == ("Exchanges, refunds", "shipping", "Charges"), "Missing descriptions use the name"
    assert escalate.options == ("true", "false")
    assert escalate.descriptions == (None, None)
    assert frustration.options == ("0", "1", "2")
    assert frustration.descriptions == ("Calm", "Frustrated", "Very angry")


def test_system_one_noul_criteria_descriptions():
    question = SystemOneRequest(
        model="lit", state="x", questions={"q": {"type": "noul", "criteria": {"true": "Spam", "false": "Not spam"}}}
    ).questions["q"]
    assert _to_decision_question("q", question).descriptions == ("Spam", "Not spam")


@pytest.mark.parametrize(
    ("probs", "expected"),
    [([1 / 3] * 3, 0.0), ([1.0, 0.0], 1.0), ([0.75, 0.25], 0.5), ([1.0], 1.0)],
)
def test_choice_confidence(probs, expected):
    assert _choice_confidence(probs) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("probs", "expected"),
    [
        ([1 / 3] * 3, 0.0),
        ([0.0, 1.0, 0.0], 1.0),
        # MAD_uniform for 3 levels is 2/3; the spread around level 1 is 0.5
        ([0.0, 0.5, 0.5], 1 - 0.5 / (2 / 3)),
        ([0.5, 0.5], 0.0),
    ],
)
def test_score_confidence(probs, expected):
    assert _score_confidence(probs) == pytest.approx(expected)
