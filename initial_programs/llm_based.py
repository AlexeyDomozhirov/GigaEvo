import os
from pathlib import Path
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from helper import load_examples, LABELS
from pydantic import BaseModel, Field

class Intent(BaseModel):
    """Classify the user intent from a Russian dialogue."""
    intent: str = Field(description="one of: greeting, capabilities, gratitude, data_catalog, pivot_table, technical, support, no_rag")
    needs_clarification: bool = Field(default=False, description="true only for data_catalog ambiguity")



SYSTEM = """Ты — классификатор интентов ассистента «Меридиан».
Выбери РОВНО ОДИН intent из списка ниже. Другие значения запрещены.

Допустимые intent (используй ровно эти строки):
greeting, capabilities, gratitude, data_catalog, pivot_table, technical, support, no_rag

Описания:
- greeting: привет/пока без задачи. «Привет!», «Пока», «Доброе утро»
- capabilities: вопрос о самом ассистенте. «Что ты умеешь?», «Кто ты?», «Ты умеешь искать витрины?», «Какие у тебя функции?»
- gratitude: спасибо, «вопросов больше нет», «мне всё ясно»
- data_catalog: найти витрину/таблицу/куб/дашборд/представление. «Найди витрины», «Покажи таблицы», «Покажи ещё» (после каталога)
- pivot_table: «построй сводную», «сгруппируй», «сделай срез», «добавь фильтр», «А за июль?» (после сводной)
- technical: объяснение, ошибка, методика, продукт. КЛАСС ПО УМОЛЧАНИЮ. «Что такое Меридиан?», «Как рассчитывается доход?», «Почему не открывается отчёт?»
- support: «позови оператора», «свяжи со специалистом», «Вариант 2» (после меню с оператором)
- no_rag: «повтори», «сократи», «переведи», «какой код ошибки я прислал»

Жёсткие правила:
- «привет»/«спасибо» + рабочая задача → верни класс задачи, не greeting/gratitude
- при сомнении → technical
- «Вариант 2» после меню с оператором → support; после меню с объяснением → technical
- «Оформи это таблицей» при наличии фактов в истории → no_rag; иначе pivot_table
- отрицание меняет интент: «не зови оператора, объясни» → technical
- needs_clarification=true ТОЛЬКО для data_catalog и ТОЛЬКО при неоднозначности «найти объект» vs «объяснить тему»

ФОРМАТ ОТВЕТА — строго JSON без markdown и пояснений:
{"intent": "<одно из 8 значений>", "needs_clarification": <true|false>}

Примеры:
История: (пусто)
Сообщение: Привет!
Ответ: {"intent": "greeting", "needs_clarification": false}

История: (пусто)
Сообщение: Что ты умеешь?
Ответ: {"intent": "capabilities", "needs_clarification": false}

История: (пусто)
Сообщение: Позови оператора.
Ответ: {"intent": "support", "needs_clarification": false}

История: A: Выберите: вариант 1 — автопроверка; вариант 2 — оператор.
Сообщение: Вариант 2.
Ответ: {"intent": "support", "needs_clarification": false}

История: A: Могу дать: вариант 1 — краткое объяснение; вариант 2 — подробный разбор.
Сообщение: Вариант 2.
Ответ: {"intent": "technical", "needs_clarification": false}

История: (пусто)
Сообщение: Привет, почему не открывается отчёт?
Ответ: {"intent": "technical", "needs_clarification": false}

История: (пусто)
Сообщение: Не зови оператора, объясни ошибку.
Ответ: {"intent": "technical", "needs_clarification": false}

История: (пусто)
Сообщение: Покажи информацию о таблицах Мозаики.
Ответ: {"intent": "data_catalog", "needs_clarification": true}
"""


llm = GigaChat(
    credentials=Path("/home/alexei/python/math/gigaevo-core/key").read_text(encoding="utf-8").replace("\n", ""),
    verify_ssl_certs=False,
    model="GigaChat-3-Lightning",
    temperature=0,
)

chain = ChatPromptTemplate.from_messages([
    ("system", "Classify the user intent. Reply with JSON."),
    ("human", "History:\n{history}\n\nMessage: {text}"),
]) | llm.with_structured_output(Intent)


def _predict_one(history, text):
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in (history or []))
    try:
        r = chain.invoke({"history": hist or "(empty)", "text": text})
        print("RAWdf:", repr(r))
        intent = r.intent if r.intent in LABELS else "technical"
        clar = r.needs_clarification and intent == "data_catalog"
        return {"intent": intent, "needs_clarification": clar}
    except Exception:
        print('df')
        return {"intent": "technical", "needs_clarification": False}


def entrypoint():
    return {r["id"]: _predict_one(r.get("history"), r.get("text")) for r in load_examples()}
