import json
from langchain_gigachat import GigaChat
import os

LABELS = ["greeting", "capabilities", "gratitude", "data_catalog",
          "pivot_table", "technical", "support", "no_rag"]

def load_examples():
    path="/home/alexei/python/math/gigaevo-core/problems/classification/examples.json"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_assistant(history):
    for m in reversed(history or []):
        if m.get("role") == "assistant":
            return (m.get("content") or "").lower()
    return ""

def get_credentials():
    """Читает GIGACHAT_CREDENTIALS из переменной окружения.

    Программа в песочнице вызывает эту функцию и не импортирует os.
    """
    key = os.environ.get("GIGACHAT_CREDENTIALS", "")
    if not key:
        raise RuntimeError(
            "GIGACHAT_CREDENTIALS is not set. "
            "Export it in the shell before running the program."
        )
    return key

def get_llm():
    return GigaChat(
        credentials=os.environ.get("GIGACHAT_CREDENTIALS", ""),
        verify_ssl_certs=False,
        model="GigaChat-3-Lightning",
        temperature=0,
        max_tokens=80,
    )

def esc_braces(s: str) -> str:
    """Экранирует { и } для f-string шаблонов LangChain.
    {{ -> { и }} -> } восстанавливаются при форматировании."""
    return s.replace("{", "{{").replace("}", "}}")
