# baseline.py
from helper import load_examples

RULES = [
    ("greeting",     ["привет", "здравств", "добрый", "пока"]),
    ("gratitude",    ["спасибо", "благодар"]),
    ("support",      ["оператор", "специалист"]),
    ("capabilities", ["умеешь", "функции", "возможности"]),
    ("pivot_table",  ["сводн", "построй"]),
    ("no_rag",       ["повтори", "сократи", "переведи"]),
    ("data_catalog", ["витрин", "таблиц", "куб"]),
]


def _predict_one(history, text):
    t = (text or "").lower()
    for intent, keys in RULES:
        if any(k in t for k in keys):
            return {"intent": intent, "needs_clarification": False}
    return {"intent": "technical", "needs_clarification": False}


def entrypoint():
    rows = load_examples()
    return {
        r["id"]: _predict_one(r.get("history") or [], r.get("text") or "")
        for r in rows
    }
