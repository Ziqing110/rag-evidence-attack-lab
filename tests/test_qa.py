from types import SimpleNamespace

import pytest

from rag_eval_lab.qa import (
    build_default_messages,
    call_qa_model,
    safe_call_qa_model,
)


class _FakeResponses:
    def __init__(self, output_text):
        self._output_text = output_text

    def create(self, **kwargs):
        return SimpleNamespace(output_text=self._output_text)


class _FakeClient:
    def __init__(self, output_text):
        self.responses = _FakeResponses(output_text=output_text)


def test_build_default_messages():
    s, u = build_default_messages("Q?", "C")
    assert "ONLY the provided context" in s
    assert "Question: Q?" in u


def test_call_qa_model_empty_context_returns_not_found():
    c = _FakeClient("unused")
    assert call_qa_model("Q", "", c, "gpt-4.1") == "NOT_FOUND"


def test_call_qa_model_normalizes_not_found_variants():
    c = _FakeClient("not found")
    assert call_qa_model("Q", "ctx", c, "gpt-4.1") == "NOT_FOUND"


def test_call_qa_model_returns_text():
    c = _FakeClient("Some answer")
    assert call_qa_model("Q", "ctx", c, "gpt-4.1") == "Some answer"


def test_safe_call_retries_and_falls_back():
    class _BadClient:
        class responses:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("boom")

    out = safe_call_qa_model("Q", "ctx", _BadClient(), "gpt-4.1", max_retries=2, sleep_s=0)
    assert out == "NOT_FOUND"

