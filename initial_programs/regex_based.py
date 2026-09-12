# regex_based.py
"""Soft regex baseline for 8-class dialogue intent classification.

Deliberately small: one short regex per class, no context resolution,
no negation handling. Anything not matched falls through to `technical`.
This is a starting point for evolution, not a finished classifier.
"""
import re
from helper import load_examples


GREETING_RX = re.compile(r"(?:привет|здравств|добрый|доброе|свидания|увидимся)")
GRATITUDE_RX = re.compile(r"(?:спасибо|благодар|спс|помогло)")
CAPABILITIES_RX = re.compile(r"(?:ты умеешь|твои (?:функции|возможности|навыки)|"
                             r"кто ты|о себе|перечисли)")
SUPPORT_RX = re.compile(r"(?:оператор|специалист|поддержк|живой человек)")
CATALOG_RX = re.compile(r"(?:витрин|таблиц|куб|дашборд|объект данных|"
                        r"набор данных|покажи ещё)")
PIVOT_RX = re.compile(r"(?:сводн|построй|сгруппируй|срез|кросс)")
NORAG_RX = re.compile(r"(?:повтори|сократи|переведи|выпиши|удали|"
                      r"перефразируй|резюме разговора)")


def _predict_one(history, text):
    t = (text or "").lower()

    if SUPPORT_RX.search(t):
        return {"intent": "support", "needs_clarification": False}

    if PIVOT_RX.search(t):
        return {"intent": "pivot_table", "needs_clarification": False}

    if CATALOG_RX.search(t):
        clar = bool(
            re.search(r"(?:информац|сведени)", t)
            and not re.search(r"(?:найди|покажи|выведи|где)", t)
        )
        return {"intent": "data_catalog", "needs_clarification": clar}

    if CAPABILITIES_RX.search(t):
        return {"intent": "capabilities", "needs_clarification": False}

    if NORAG_RX.search(t):
        return {"intent": "no_rag", "needs_clarification": False}

    if GRATITUDE_RX.search(t):
        return {"intent": "gratitude", "needs_clarification": False}

    if GREETING_RX.search(t):
        return {"intent": "greeting", "needs_clarification": False}

    return {"intent": "technical", "needs_clarification": False}


def entrypoint():
    rows = load_examples()
    return {
        r["id"]: _predict_one(r.get("history") or [], r.get("text") or "")
        for r in rows
    }
