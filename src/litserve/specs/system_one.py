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
import asyncio
import inspect
import logging
import math
import time
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from litserve.callbacks.base import EventTypes
from litserve.constants import _DEFAULT_LIT_API_PATH
from litserve.specs.base import LitSpec
from litserve.utils import LitAPIStatus, ResponseBufferItem

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from litserve import LitAPI, LitServer

JSONContent = Union[str, dict[str, Any], list[Any]]
NOUL_OPTIONS = ("true", "false")
MAX_SCORE_LEVELS = 10


class _BaseQuestion(BaseModel):
    # unknown keys inside a question are rejected: a typo there would silently change the question
    model_config = ConfigDict(extra="forbid")

    instructions: Optional[JSONContent] = None


class NoulQuestion(_BaseQuestion):
    type: Literal["noul"]
    criteria: Optional[dict[Literal["true", "false"], Optional[JSONContent]]] = None

    @model_validator(mode="after")
    def _check_instructions_or_criteria(self):
        if self.instructions is None and self.criteria is None:
            raise ValueError("a noul question needs `instructions` or `criteria`")
        return self


class ChoiceQuestion(_BaseQuestion):
    type: Literal["choice"]
    criteria: dict[str, Optional[JSONContent]] = Field(min_length=1)


class ScoreQuestion(_BaseQuestion):
    type: Literal["score"]
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_SCORE_LEVELS)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    # unknown top-level keys are ignored, so fields added by gateways (provider, session_id, ...) pass through
    model: str
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, JSONContent]
    probabilities: dict[str, float]
    confidence: float


class SystemOneUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]]
    usage: SystemOneUsage


@dataclass(frozen=True)
class DecisionQuestion:
    """A single question from the request.

    Every question type is a list of options to score, so ``predict`` handles all three the same way.

    Attributes:
        id: The name the client gave the question. Use it as the key in the dict returned by ``predict``.
        type: ``"noul"`` (yes/no), ``"choice"`` (pick one option) or ``"score"`` (rate on ordered levels).
        instructions: What to decide, as sent by the client (text, a JSON object, a list, or None).
        options: The options to score, in order:

            - choice: the option names, e.g. ``("billing", "technical")``
            - score: the levels from low to high, ``("0", "1", ..., "k-1")``
            - noul: ``("true", "false")``

        descriptions: The client's description of each option, aligned with ``options``. A choice option sent
            without a description is described by its name; a noul without criteria has ``None`` descriptions.

    """

    id: str
    type: Literal["noul", "choice", "score"]
    instructions: Optional[JSONContent]
    options: tuple[str, ...]
    descriptions: tuple[Optional[JSONContent], ...]


@dataclass(frozen=True)
class DecisionRequest:
    """The input your ``predict`` receives.

    Attributes:
        model: The model name sent by the client.
        state: The content every question is about (text, a JSON object, or a list).
        questions: The questions to answer, in request order. Answer each one independently.

    """

    model: str
    state: JSONContent
    questions: tuple[DecisionQuestion, ...]


@dataclass
class DecisionOutput:
    """Return this from ``predict`` instead of a plain dict to report token usage.

    Attributes:
        logits: Same as the plain return value: ``{question.id: logits}``.
        input_tokens: Tokens read to answer the request, reported back as ``usage.input_tokens``.
        output_tokens: Reported back as ``usage.output_tokens``.

    Example:
        >>> DecisionOutput({"is_spam": [2.3, -1.0]}, input_tokens=42)  # doctest: +SKIP

    """

    logits: Mapping[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_ONE_API_EXAMPLE = """
Please follow the example below for guidance on how to use the System One spec:

```python
import litserve as ls
from litserve.specs import DecisionRequest, SystemOneSpec

class DecisionAPI(ls.LitAPI):
    def setup(self, device):
        self.model = load_decision_model(device)

    def predict(self, request: DecisionRequest):
        # one logit per option: a list aligned with `question.options`, or a dict keyed by option
        return {q.id: self.model.score(request.state, q.instructions, q.descriptions) for q in request.questions}

if __name__ == "__main__":
    server = ls.LitServer(DecisionAPI(spec=SystemOneSpec()))
    server.run()
```
"""


def _to_decision_question(qid: str, question: Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]) -> DecisionQuestion:
    if question.type == "noul":
        criteria = question.criteria or {}
        options, descriptions = NOUL_OPTIONS, tuple(criteria.get(option) for option in NOUL_OPTIONS)
    elif question.type == "choice":
        options = tuple(question.criteria)
        descriptions = tuple(name if desc is None else desc for name, desc in question.criteria.items())
    else:
        options, descriptions = tuple(str(level) for level in range(len(question.criteria))), tuple(question.criteria)
    return DecisionQuestion(qid, question.type, question.instructions, options, descriptions)


def _to_float_list(question: DecisionQuestion, logits: Any) -> list[float]:
    if isinstance(logits, Mapping):
        if set(logits) != set(question.options):
            raise ValueError(f"expected logits for options {list(question.options)}, got {list(logits)}")
        values = [logits[option] for option in question.options]
    else:
        values = list(logits)
        if len(values) != len(question.options):
            raise ValueError(f"expected {len(question.options)} logits, got {len(values)}")
    values = [float(v) for v in values]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("logits must be finite numbers")
    return values


def _softmax(logits: list[float], temperature: float) -> list[float]:
    peak = max(logits)
    exps = [math.exp((x - peak) / temperature) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _choice_confidence(probs: list[float]) -> float:
    # TypeSafe's formula: how far the top probability is above an even split, (p_max - 1/n) / (1 - 1/n)
    n = len(probs)
    if n == 1:
        return 1.0
    return (max(probs) - 1 / n) / (1 - 1 / n)


def _score_confidence(probs: list[float]) -> float:
    # TypeSafe's formula: spread around the most likely level m, relative to a uniform distribution's
    # spread around the middle level, max(0, 1 - sum(p_i * |i - m|) / MAD_uniform)
    n = len(probs)
    m = probs.index(max(probs))
    spread = sum(p * abs(i - m) for i, p in enumerate(probs))
    mad_uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / mad_uniform)


def _to_answer(question: DecisionQuestion, logits: Any, temperature: float):
    probs = _softmax(_to_float_list(question, logits), temperature)
    if question.type == "noul":
        return NoulAnswer(noul=probs[0])

    probabilities = dict(zip(question.options, probs))
    if question.type == "choice":
        return ChoiceAnswer(
            choice=question.options[probs.index(max(probs))],
            probabilities=probabilities,
            confidence=_choice_confidence(probs),
        )
    return ScoreAnswer(
        score=sum(i * p for i, p in enumerate(probs)),
        legend=dict(zip(question.options, question.descriptions)),
        probabilities=probabilities,
        confidence=_score_confidence(probs),
    )


def _to_python(logits: Any) -> Any:
    # numpy arrays and torch tensors become plain lists, so the response can be sent back from the worker
    if hasattr(logits, "tolist"):
        return logits.tolist()
    if isinstance(logits, Mapping):
        return {str(k): float(v.item() if hasattr(v, "item") else v) for k, v in logits.items()}
    return [float(v.item() if hasattr(v, "item") else v) for v in logits]


class SystemOneSpec(LitSpec):
    """Serve a decision model with the System One API (``POST /v1/systemone``).

    System One is the API of TypeSafe's Jev, also served by Kev, SGLang, llama.cpp and Ollama. A client sends some
    ``state`` and named questions; the server answers each one with probabilities instead of generated text. The
    official TypeSafe SDKs work with a LitServe server by changing only the base URL.

    Your ``predict`` receives a :class:`DecisionRequest` and returns one logit per option for each question, as
    ``{question.id: logits}``. Logits can be a list aligned with ``question.options``, a dict keyed by option, or a
    numpy array or torch tensor. The spec does the rest: it validates the request, applies ``temperature`` and softmax,
    and returns typed answers with ``confidence`` computed by TypeSafe's published formulas. Return a
    :class:`DecisionOutput` instead to also report token usage.

    Args:
        temperature: Logits are divided by this before the softmax. Values above 1 make answers less confident. Fit it
            on labelled data to calibrate your model. Defaults to 1.0.
        max_questions: Requests with more questions are rejected with a 422 error. Defaults to 64.
        max_options: Choice questions with more options are rejected with a 422 error. Defaults to 255.

    Example:
        ```python
        import litserve as ls
        from litserve.specs import DecisionRequest, SystemOneSpec

        class DecisionAPI(ls.LitAPI):
            def setup(self, device):
                self.model = load_decision_model(device)

            def predict(self, request: DecisionRequest):
                return {q.id: self.model.score(request.state, q) for q in request.questions}

        server = ls.LitServer(DecisionAPI(spec=SystemOneSpec(temperature=1.5)))
        server.run()
        ```

        Then query it with the TypeSafe SDK:

        ```python
        from typesafe_sdk import Choice, Noul, TypeSafeClient

        client = TypeSafeClient(api_key="lit", base_url="http://127.0.0.1:8000", model="my-model")
        response = client.system_one(
            state="I was charged twice for my order.",
            questions={
                "team": Choice(criteria={"billing": "Payments and refunds", "technical": "Bugs and outages"}),
                "urgent": Noul(instructions="Does this need a reply today?"),
            },
        )
        response.answers["team"].choice  # "billing"
        ```

    """

    def __init__(self, *, temperature: float = 1.0, max_questions: int = 64, max_options: int = 255):
        super().__init__()
        if not temperature > 0:
            raise ValueError(f"temperature must be greater than 0, got {temperature}")
        self.api_path = "/v1/systemone"  # default api path
        self.temperature = temperature
        self.max_questions = max_questions
        self.max_options = max_options

    def pre_setup(self, lit_api: "LitAPI"):
        from litserve import LitAPI

        # Override the spec's api_path only if provided
        if lit_api._api_path and lit_api._api_path not in (_DEFAULT_LIT_API_PATH, self.api_path):
            self.api_path = lit_api._api_path
            warnings.warn(
                f"Custom API path detected: '{self.api_path}'. "
                "System One SDKs only call the default path '/v1/systemone'. "
                f"To use '{self.api_path}', send HTTP requests directly or use a client that supports custom endpoints."
            )

        self.add_endpoint(self.api_path, self.system_one_endpoint, ["POST"])

        # validate LitAPI methods
        is_encode_response_original = lit_api.encode_response.__code__ is LitAPI.encode_response.__code__
        if inspect.isgeneratorfunction(lit_api.predict) or (
            not is_encode_response_original and inspect.isgeneratorfunction(lit_api.encode_response)
        ):
            raise ValueError(
                "You are using yield in your predict or encode_response method, which is used for streaming. "
                "SystemOneSpec doesn't support streaming because a decision is a single readout. "
                "Please replace yield with return.\n" + SYSTEM_ONE_API_EXAMPLE
            )

    def setup(self, server: "LitServer"):
        super().setup(server)
        logger.info("System One Spec is ready.")

    def decode_request(self, request: SystemOneRequest, context_kwargs: Optional[dict] = None) -> DecisionRequest:
        questions = tuple(_to_decision_question(qid, question) for qid, question in request.questions.items())
        return DecisionRequest(model=request.model, state=request.state, questions=questions)

    def encode_response(self, output: Any, context_kwargs: Optional[dict] = None) -> dict:
        if isinstance(output, Mapping):
            output = DecisionOutput(logits=output)
        if not isinstance(output, DecisionOutput):
            raise ValueError(
                f"Expected predict to return a dict of question id to logits, or a DecisionOutput, "
                f"but got type {type(output)}.\n{SYSTEM_ONE_API_EXAMPLE}"
            )
        return {
            "logits": {qid: _to_python(logits) for qid, logits in output.logits.items()},
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
        }

    def _validate_limits(self, request: SystemOneRequest) -> None:
        # errors use the same shape as FastAPI's own 422 responses, which System One clients parse
        errors = []
        if len(request.questions) > self.max_questions:
            msg = f"At most {self.max_questions} questions are allowed, got {len(request.questions)}"
            errors.append({"loc": ["body", "questions"], "msg": msg, "type": "too_long"})
        for qid, question in request.questions.items():
            if question.type == "choice" and len(question.criteria) > self.max_options:
                msg = f"At most {self.max_options} options are allowed, got {len(question.criteria)}"
                errors.append({"loc": ["body", "questions", qid, "choice", "criteria"], "msg": msg, "type": "too_long"})
        if errors:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=errors)

    def _build_response(self, request: SystemOneRequest, output: dict) -> SystemOneResponse:
        logits = output["logits"]
        if unknown := set(logits) - set(request.questions):
            raise ValueError(f"predict returned logits for unknown questions {sorted(unknown)}")

        answers = {}
        for qid, question in request.questions.items():
            if qid not in logits:
                raise ValueError(f"predict returned no logits for question '{qid}'")
            try:
                answers[qid] = _to_answer(_to_decision_question(qid, question), logits[qid], self.temperature)
            except (TypeError, ValueError) as e:
                raise ValueError(f"invalid logits for question '{qid}': {e}") from e

        usage = SystemOneUsage(input_tokens=output["input_tokens"], output_tokens=output["output_tokens"])
        return SystemOneResponse(model=request.model, answers=answers, usage=usage)

    async def system_one_endpoint(self, request: SystemOneRequest) -> SystemOneResponse:
        self._validate_limits(request)

        logger.debug("Received System One request: %s", request)
        uid = uuid.uuid4()
        event = asyncio.Event()
        self.response_buffer[uid] = ResponseBufferItem(event=event)

        # Trigger callback
        self._server._callback_runner.trigger_event(
            EventTypes.ON_REQUEST.value,
            active_requests=self._server.active_requests,
            litserver=self._server,
        )

        self.request_queue.put_nowait((self.response_queue_id, uid, time.monotonic(), request.model_copy()))
        await event.wait()

        response, response_status = self.response_buffer.pop(uid).response

        if response_status == LitAPIStatus.ERROR and isinstance(response, HTTPException):
            logger.error("Error in System One request: %s", response)
            raise response

        if response_status == LitAPIStatus.ERROR:
            logger.error("Error in System One request: %s", response)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            return self._build_response(request, response)
        except ValueError as e:
            logger.error("Invalid output from predict: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
