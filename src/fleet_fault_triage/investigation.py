"""One bounded Foundry agent selects approved checks; numeric facts stay deterministic."""

import asyncio
import inspect
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from fleet_fault_triage.contracts import SCOPE, Brief, Contract, canonical

CASE_TIMEOUT_SECONDS = 90

MISSING = {
    "timing": "Measurement-to-failure timing is unavailable.",
    "semantics": "Predictor meanings are anonymised; physical diagnosis is unsupported.",
    "field": "No mine-site or independent field validation has been performed.",
    "quality": "The input failed the declared missingness quality policy.",
}


class Selection(Contract):
    case_id: str
    status: Literal["draft", "incomplete"]
    citations: list[str] = Field(max_length=5)
    missing_evidence: list[Literal["timing", "semantics", "field", "quality"]]


def deterministic_brief(case):
    return Brief(
        case_id=case.case_id,
        release_sha256=case.release_sha256,
        facts=[f"Recorded case {case.record_id} follows route {case.route}.", SCOPE],
        estimates=[]
        if case.probability is None
        else [
            f"APS benchmark probability estimate {case.probability!r}; threshold {case.threshold!r}."
        ],
        missing_evidence=[MISSING["timing"], MISSING["semantics"], MISSING["field"]]
        + ([MISSING["quality"]] if case.probability is None else []),
        proposed_checks=[],
        citations=[],
        origin="deterministic_report",
    )


class Investigator:
    def __init__(self, store, release, corpus, retrieval, allowed_case_ids, *, guard=None):
        self.store, self.release, self.corpus, self.retrieval = store, release, corpus, retrieval
        self.allowed = frozenset(allowed_case_ids)
        self.trace = []
        self.guard = guard
        self.active_case = None
        self.retrieved_ids = set()
        self.deadline = float("inf")

    def get_case_evidence(self, case_id: str) -> dict:
        """Read the authorised case's exact immutable score and known limitations."""
        if case_id not in self.allowed:
            raise PermissionError("case outside the read-only investigation scope")
        if self.active_case is not None and case_id != self.active_case:
            raise PermissionError("tool cannot cross the current case boundary")
        case = self.store.case(case_id)
        if case.release_sha256 != self.release.sha256:
            raise ValueError("stale case release")
        return deterministic_brief(case).model_dump()

    def get_model_card(self, version: str) -> dict:
        """Read the exact pinned model identity and benchmark scope; no alias resolution."""
        if version != self.release.model_uri:
            raise PermissionError("model outside the release scope")
        return {"model_uri": version, "release_sha256": self.release.sha256, "scope": SCOPE}

    async def search_runbooks(self, query: str, corpus_version: str, top_k: int = 5) -> list[dict]:
        """Retrieve at most five approved project runbooks; never browse arbitrary URLs/files."""
        if corpus_version != self.release.corpus_sha256:
            raise ValueError("retrieval corpus differs from the frozen release")
        # Keep synchronous I/O off the event loop; the runtime adapter supplies
        # transport timeouts from the remaining case deadline.
        rows = await asyncio.to_thread(self.retrieval, query, corpus_version, top_k)
        for row in rows:
            self.corpus.resolve(row["id"], corpus_version)
            self.retrieved_ids.add(row["id"])
        return rows

    def assemble(self, selection, case_id, *, origin="foundry"):
        selection = Selection.model_validate(selection)
        if selection.case_id != case_id:
            raise ValueError("assistant changed the case identity")
        base = self.get_case_evidence(case_id)
        documents = [
            self.corpus.resolve(c, self.release.corpus_sha256) for c in selection.citations
        ]
        if len(set(selection.citations)) != len(documents):
            raise ValueError("duplicate citations")
        if selection.status == "draft" and not documents:
            raise ValueError("a completed brief must cite an approved runbook")
        if origin == "foundry" and (
            self.active_case != case_id or not set(selection.citations) <= self.retrieved_ids
        ):
            raise ValueError("citation was not retrieved for this case/invocation")
        base.update(
            status=selection.status,
            citations=selection.citations,
            proposed_checks=[d["check"] for d in documents],
            origin=origin,
            failure_reason="insufficient_supported_evidence"
            if selection.status == "incomplete"
            else None,
        )
        # The model cannot replace facts, numbers, uncertainty or write human review state.
        return Brief.model_validate(base)

    async def run(self, case_id, question, client, prompt: str):
        from agent_framework import Agent, ChatContext, chat_middleware

        self.active_case = None
        self.get_case_evidence(case_id)
        self.active_case, self.retrieved_ids = case_id, set()
        if self.release.origin == "azure" and self.guard is None:
            raise ValueError("hosted inference requires the approved cost/request guard")
        if len(question.encode()) > 2000:
            raise ValueError("question exceeds the bounded input contract")
        calls, input_bound, tool_calls = 0, 0, 1

        @chat_middleware
        async def bound_chat(context: ChatContext, call_next):
            nonlocal calls, input_bound
            # UTF-8 bytes bound token count conservatively for the chosen text-only inputs.
            request_bound = len(canonical([m.to_dict() for m in context.messages]).encode())
            request_bound += (
                len(canonical(["get_case_evidence", "get_model_card", "search_runbooks"]).encode())
                + 2500
            )
            if calls >= 3 or input_bound + request_bound > 12_000:
                raise ValueError("model-turn/input budget exhausted")
            calls += 1
            if self.guard:
                self.guard("inference")
            await call_next()
            usage = context.result.usage_details
            if (
                not usage
                or usage.get("input_token_count") is None
                or usage.get("output_token_count") is None
            ):
                raise ValueError("provider omitted token usage")
            if usage["input_token_count"] > request_bound or usage["output_token_count"] > 1500:
                raise ValueError("provider exceeded the declared token bound")
            input_bound += usage["input_token_count"]
            self.trace.append(
                {
                    "turn": calls,
                    "response_id": context.result.response_id,
                    "usage": dict(usage),
                    "input_upper_bound": request_bound,
                }
            )

        def bounded(function):
            # Preserve typed tool signatures using functools.wraps for framework introspection.
            import functools

            @functools.wraps(function)
            async def wrapped(*args, **kwargs):
                nonlocal tool_calls
                if tool_calls >= 6:
                    raise ValueError("tool-call budget exhausted")
                tool_calls += 1
                result = (
                    await function(*args, **kwargs)
                    if inspect.iscoroutinefunction(function)
                    else await asyncio.to_thread(function, *args, **kwargs)
                )
                if len(canonical(result).encode()) > 6000:
                    raise ValueError("tool output exceeds bound")
                self.trace.append({"tool": function.__name__, "call": tool_calls})
                return result

            return wrapped

        agent = Agent(
            client=client,
            instructions=prompt,
            tools=[
                bounded(self.get_case_evidence),
                bounded(self.get_model_card),
                bounded(self.search_runbooks),
            ],
            default_options={"max_tokens": 1500, "response_format": Selection, "store": False},
            middleware=[bound_chat],
        )
        payload = {
            "case_id": case_id,
            "question": question,
            "model_uri": self.release.model_uri,
            "corpus_version": self.corpus.version,
        }
        try:
            self.deadline = time.monotonic() + CASE_TIMEOUT_SECONDS
            async with asyncio.timeout(CASE_TIMEOUT_SECONDS):
                documents = await self.search_runbooks(question[:1000], self.corpus.version, 5)
                if len(canonical(documents).encode()) > 6000:
                    raise ValueError("retrieval context exceeds bound")
                payload["retrieved_runbooks"] = documents
                self.trace.append({"tool": "search_runbooks", "call": 1, "case_id": case_id})
                response = await agent.run(canonical(payload))
            selection = response.value
            if selection is None:
                selection = json.loads(response.text)
            return self.assemble(selection, case_id)
        except Exception as error:
            self.trace.append({"failure_type": type(error).__name__})
            base = self.get_case_evidence(case_id)
            base.update(
                status="incomplete",
                origin="foundry",
                failure_reason="budget_policy_or_provider_failure",
            )
            return Brief.model_validate(base)


def prompt_text(root: Path):
    return (root / "resources/prompts/investigator.md").read_text()
