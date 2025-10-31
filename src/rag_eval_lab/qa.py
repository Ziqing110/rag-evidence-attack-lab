import os
import time
from typing import Any

from dotenv import load_dotenv


def build_default_messages(question: str, context: str) -> tuple[str, str]:
    """
    Build system/user messages for context-grounded QA.
    """
    system = (
        "You are a question-answering assistant. "
        "Answer the question using ONLY the provided context. "
        "If the answer is not explicitly supported by the context, reply exactly: NOT_FOUND. "
        "Keep the answer concise."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return system, user


def call_qa_model(
    question: str,
    context: str,
    client: Any,
    model: str,
    max_tokens: int = 256,
) -> str:
    """
    Call an OpenAI Responses model with context-grounded QA prompting.
    """
    context = (context or "").strip()
    if not context:
        return "NOT_FOUND"

    system, user = build_default_messages(question, context)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_output_tokens=max_tokens,
    )

    text = (getattr(resp, "output_text", "") or "").strip()
    if not text:
        return "NOT_FOUND"

    if text.lower() in {"unanswerable", "not found", "not_found"}:
        return "NOT_FOUND"
    return text


def safe_call_qa_model(
    question: str,
    context: str,
    client: Any,
    model: str,
    max_retries: int = 3,
    sleep_s: float = 1.0,
) -> str:
    """
    Retry wrapper around call_qa_model.
    """
    for attempt in range(max_retries):
        try:
            return call_qa_model(question=question, context=context, client=client, model=model)
        except Exception:
            time.sleep(sleep_s * (attempt + 1))
    return "NOT_FOUND"


def build_openai_client_from_env() -> tuple[Any, str]:
    """
    Build OpenAI client and resolve model name from environment.
    """
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    return client, model

