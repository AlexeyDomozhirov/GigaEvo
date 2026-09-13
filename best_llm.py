from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from helper import load_examples, LABELS, get_credentials, esc_braces, get_llm

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
- Одна и та же короткая фраза может означать разные интенты в зависимости от последней реплики ассистента. Всегда смотри на историю перед выбором intent.
- «привет»/«спасибо» + рабочая задача → верни класс задачи, не greeting/gratitude
- при сомнении → technical
- «Вариант 2» после меню с оператором → support; после меню с объяснением → technical
- «Оформи это таблицей» при наличии фактов в истории → no_rag; иначе pivot_table
- отрицание меняет интент: «не зови оператора, объясни» → technical; «не повторяй инструкцию, лучше позови» → support
- needs_clarification=true ТОЛЬКО для data_catalog и ТОЛЬКО при неоднозначности «найти объект» vs «объяснить тему»

ФОРМАТ ОТВЕТА — строго JSON без markdown и пояснений:
{"intent": "<одно из 8 значений>", "needs_clarification": <true|false>}

# Алгоритм самопроверки перед ответом (без вывода шагов):
# 1) Если есть явный запрос на построение чисел («построй», «срез», «фильтр») -> pivot_table.
# 2) Если есть явный поиск объекта каталога («найди», «где лежат», «покажи таблицы») -> data_catalog.
# 3) Если все факты уже в истории («повтори», «сократи», «какой код ошибки») -> no_rag.
# 4) Если пользователь выбирает пункт из меню поддержки -> support; из меню объяснения -> technical.
# 5) Отрицание переворачивает смысл: «не зови...» -> technical; «не повторяй..., позови» -> support.
# 6) При любой оставшейся неопределённости -> technical.
# Проверь флаг: needs_clarification=true только если query можно понять И как поиск объекта, И как объяснение темы.

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

История: A: Могу дать: вариант 1 — краткое объяснение; вариант 2 — подробный разбор условий.
Сообщение: Вариант 2.
Ответ: {"intent": "technical", "needs_clarification": false}

История: (пусто)
Сообщение: Привет, почему не открывается отчёт?
Ответ: {"intent": "technical", "needs_clarification": false}

История: (пусто)
Сообщение: Не зови оператора, объясни ошибку CHECKSUM_MISMATCH.
Ответ: {"intent": "technical", "needs_clarification": false}

История: (пусто)
Сообщение: Покажи информацию о таблицах Мозаики.
Ответ: {"intent": "data_catalog", "needs_clarification": true}

# Context-pair fixes
История: A: Выберите: вариант 1 — автопроверка; вариант 2 — оператор.
Сообщение: Вариант 2.
Ответ: {"intent": "support", "needs_clarification": false}

История: A: Могу дать: вариант 1 — краткое объяснение; вариант 2 — подробный разбор условий.
Сообщение: Вариант 2.
Ответ: {"intent": "technical", "needs_clarification": false}

История: A: Найдены PrismDaily и PrismDeals. Есть ещё объекты каталога.
Сообщение: Покажи ещё.
Ответ: {"intent": "data_catalog", "needs_clarification": false}

История: A: Мы обсудили порог. Могу показать обработку возвратов.
Сообщение: Покажи ещё.
Ответ: {"intent": "technical", "needs_clarification": false}

История: A: Исходный объём — размер операции. Чистый — минус возврат.
Сообщение: Оформи это таблицей.
Ответ: {"intent": "no_rag", "needs_clarification": false}

История: A: Могу получить числа и построить сводную по указанным параметрам.
Сообщение: Оформи это таблицей.
Ответ: {"intent": "pivot_table", "needs_clarification": false}

История: A: Методика V2 Искры начинает действовать 1 апреля 2026 года.
Сообщение: Когда она начинает действовать?
Ответ: {"intent": "no_rag", "needs_clarification": false}

История: A: У Искры есть методика V2 с обновлёнными правилами активности.
Сообщение: Когда она начинает действовать?
Ответ: {"intent": "technical", "needs_clarification": false}

История: A: За август: внутренний маршрут — 1200 АРК, межплатформенный — 800 АРК.
Сообщение: А за июль?
Ответ: {"intent": "pivot_table", "needs_clarification": false}

# Negations
История: (пусто)
Сообщение: Не зови оператора, объясни ошибку CHECKSUM_MISMATCH.
Ответ: {"intent": "technical", "needs_clarification": false}

История: (пусто)
Сообщение: Специалиста не подключайте. Что делать, если watermark не обновился?
Ответ: {"intent": "technical", "needs_clarification": false}

История: (пусто)
Сообщение: Не повторяй инструкцию, лучше позови оператора.
Ответ: {"intent": "support", "needs_clarification": false}

# No_rag facts about dialogue
История: A: В логе вашего примера написано REVISION_CONFLICT.
Сообщение: Какой код ошибки я прислал?
Ответ: {"intent": "no_rag", "needs_clarification": false}

История: A: Обсудим Зонтик.
Сообщение: Про какой продукт мы говорили?
Ответ: {"intent": "no_rag", "needs_clarification": false}
"""


def _serialize_history(history):
    """
    Serialize up to 7 last turns with role labels, prioritizing recent context.
    Prevents token overflow while preserving disambiguating signals.
    """
    if not history:
        return "(empty)"
    # Keep at most 3 latest user turns and 4 latest assistant turns
    kept = []
    u_count, a_count = 0, 0
    for turn in reversed(history or []):
        r = turn.get("role")
        c = turn.get("content", "")
        if r == "user" and u_count < 3:
            kept.append(f"{r}: {c}")
            u_count += 1
        elif r == "assistant" and a_count < 4:
            kept.append(f"{r}: {c}")
            a_count += 1
    kept.reverse()
    return "\n".join(kept) if kept else "(empty)"


llm = get_llm()#DONT CHANGE THIS!!!

chain = ChatPromptTemplate.from_messages([
    ("system", esc_braces(SYSTEM)),
    ("human", "History:\n{history}\n\nMessage: {text}"),
]) | llm.with_structured_output(Intent)


def _predict_one(history, text):
    hist = _serialize_history(history)
    try:
        r = chain.invoke({"history": hist, "text": text})
        intent = r.intent if r.intent in LABELS else "technical"
        clar = r.needs_clarification if intent == "data_catalog" else False
        return {"intent": intent, "needs_clarification": clar}
    except Exception:
        return {"intent": "technical", "needs_clarification": False}


def entrypoint():
    return {r["id"]: _predict_one(r.get("history"), r.get("text")) for r in load_examples()}
