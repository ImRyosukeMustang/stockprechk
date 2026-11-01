"""BigData research-agent client used by the dashboard chatbot."""

from __future__ import annotations

import json
from typing import Any

import config

log = config.get_logger(__name__)
API_URL = "https://agents.bigdata.com/v1/research-agent"


def _as_citations(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _read_event(payload: Any) -> tuple[str | None, list[Any], str | None]:
    """Extract answer text, citations, and a possible streaming delta."""
    if isinstance(payload, str):
        return None, [], payload
    if not isinstance(payload, dict):
        return None, [], None

    event_type = str(payload.get("type", "")).lower()
    answer = None
    citations: list[Any] = []
    delta = None
    for key in ("final_answer", "answer", "response", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            if "delta" in event_type or event_type in {"token", "text_chunk"}:
                delta = (delta or "") + value
            else:
                answer = value
    for key in ("citations", "sources", "references"):
        if key in payload:
            citations.extend(_as_citations(payload[key]))

    nested = payload.get("data") or payload.get("result")
    if isinstance(nested, (dict, list)):
        children = nested if isinstance(nested, list) else [nested]
        for child in children:
            child_answer, child_citations, child_delta = _read_event(child)
            answer = child_answer or answer
            citations.extend(child_citations)
            delta = (delta or "") + (child_delta or "")
    return answer, citations, delta


def ask_bigdata(question: str, research_effort: str = "lite") -> dict:
    """Ask BigData's research agent and collect its SSE answer and citations."""
    if not config.BIGDATA_API_KEY:
        return {
            "answer": None,
            "citations": [],
            "error": "BIGDATA_API_KEY is not configured.",
        }
    if not question or not question.strip():
        return {"answer": None, "citations": [], "error": "Enter a question first."}

    try:
        import requests

        answer = None
        token_parts: list[str] = []
        citations: list[Any] = []
        with requests.post(
            API_URL,
            headers={
                "X-API-KEY": config.BIGDATA_API_KEY,
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            json={"query": question.strip(), "research_effort": research_effort},
            stream=True,
            timeout=(10, 180),
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = data
                event_answer, event_citations, delta = _read_event(payload)
                answer = event_answer or answer
                citations.extend(event_citations)
                if delta:
                    token_parts.append(delta)

        if answer is None and token_parts:
            answer = "".join(token_parts)
        if answer is None:
            return {
                "answer": None,
                "citations": citations,
                "error": "The research agent returned no answer.",
            }
        return {"answer": answer, "citations": citations, "error": None}
    except Exception as exc:
        log.warning("BigData research request failed (%s).", type(exc).__name__)
        return {
            "answer": None,
            "citations": [],
            "error": f"BigData request failed: {type(exc).__name__}.",
        }
