"""Эксперимент: сколько токенов занимают тексты у разных токенизаторов.

Две группы текстов:
  1. Один и тот же текст на английском, русском и казахском (кириллица и
     латиница) — сравнение языков и письменностей.
  2. Реальные тексты только на русском (сообщение клиента, переписка, код,
     числа и суммы) — сколько токенов занимает «живое» общение.

Токенизаторы:
  * tiktoken (OpenAI): cl100k_base и o200k_base — считаются локально;
  * Claude: официальный метод count_tokens (нужен ключ в файле .env).

Запуск:
    pip install -r requirements.txt
    copy .env.example .env      # и вставьте свой ключ ANTHROPIC_API_KEY
    python token_count.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import anthropic
import tiktoken
from dotenv import load_dotenv

# Кодировки OpenAI: cl100k_base (GPT-4 / GPT-3.5), o200k_base (GPT-4o и новее).
TIKTOKEN_ENCODINGS = ("cl100k_base", "o200k_base")

# Модели Claude. У каждой модели свой токенизатор, поэтому считаем для каждой.
CLAUDE_MODELS = (
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-2",
)

# count_tokens возвращает токены всего запроса, а не только текста: к ним
# добавляется фиксированная служебная добавка на оформление сообщения.
# Чтобы её вычесть, считаем запрос из одного символа «.» и принимаем, что сам
# символ — ровно 1 токен (приближённо, возможна ошибка в ±1 токен).
OVERHEAD_PROBE = "."

# Ширина колонки с названием текста в таблицах.
NAME_W = 28

# ---------------------------------------------------------------------------
# Группа 1. Один текст на разных языках
# ---------------------------------------------------------------------------

EN = (
    "Large language models split text into small pieces called tokens. "
    "The number of tokens affects the cost and speed of processing a request, "
    "as well as how much information fits into the context window. "
    "Different languages are split into tokens differently."
)

RU = (
    "Большие языковые модели разбивают текст на небольшие части, называемые токенами. "
    "Количество токенов влияет на стоимость и скорость обработки запроса, "
    "а также на то, сколько информации помещается в контекстное окно. "
    "Разные языки разбиваются на токены по-разному."
)

KK_CYRL = (
    "Үлкен тіл модельдері мәтінді токен деп аталатын шағын бөліктерге бөледі. "
    "Токендер саны сұранысты өңдеу құны мен жылдамдығына, "
    "сондай-ақ контекстік терезеге қанша ақпарат сиятынына әсер етеді. "
    "Әртүрлі тілдер токендерге әртүрлі бөлінеді."
)

# Соответствие казахской кириллицы и казахской латиницы (редакция 2021 г.,
# апрельская версия с «ñ» для «ң»). Источник таблицы:
# https://en.wikipedia.org/wiki/Kazakh_alphabets (столбец «2021»).
# Ключи — строчные буквы; заглавные получаются автоматически (см. _capitalize).
KK_CYRL_TO_LATN = {
    "а": "a", "ә": "ä", "б": "b", "в": "v", "г": "g", "ғ": "ğ", "д": "d",
    "е": "e", "ё": "io", "ж": "j", "з": "z", "и": "i", "й": "i", "к": "k",
    "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "ñ", "о": "o", "ө": "ö",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "ū", "ү": "ü",
    "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "tş", "ш": "ş", "щ": "ştş",
    "ъ": "", "ы": "y", "і": "ı", "ь": "", "э": "e", "ю": "iu", "я": "ia",
}


def _capitalize(latin: str) -> str:
    """Делает заглавной первую букву, учитывая турецкую пару i/İ и ı/I."""
    if not latin:
        return latin
    first = {"i": "İ", "ı": "I"}.get(latin[0], latin[0].upper())
    return first + latin[1:]


def kazakh_cyrillic_to_latin(text: str) -> str:
    """Транслитерирует казахский текст с кириллицы на латиницу (2021)."""
    out: list[str] = []
    for ch in text:
        low = ch.lower()
        if low not in KK_CYRL_TO_LATN:
            out.append(ch)  # пробелы, знаки препинания и т. п.
            continue
        latin = KK_CYRL_TO_LATN[low]
        out.append(latin if ch == low else _capitalize(latin))
    return "".join(out)


TEXTS = {
    "Английский": EN,
    "Русский": RU,
    "Казахский (кириллица)": KK_CYRL,
    "Казахский (латиница)": kazakh_cyrillic_to_latin(KK_CYRL),
}

# ---------------------------------------------------------------------------
# Группа 2. Реальные тексты (только русский, без перевода)
# ---------------------------------------------------------------------------

# Короткое сообщение клиента — приведено как есть, вместе с опечаткой
# (нет пробела после точки в «телефона.Звонить»).
CLIENT_MESSAGE = (
    "Здравствуйте! Есть ряд вопросов. Прошу позвонить по номеру телефона."
    "Звонить из Казахстана"
)

# Переписка менеджера с клиентом — как есть.
CHAT_DIALOG = (
    "Клиент: Здравствуйте, хочу узнать подробнее о семинаре [ссылка]\n"
    'Менеджер: Здравствуйте! Вас приветствует "Pro Sport Performance", '
    "подскажите чем мы можем Вам помочь?\n"
    "Менеджер: Расширенный курс по спортивной реабилитации. "
    "Ведущий: Кристофер Дженкинс. Даты: 19 - 23 октября. "
    "Время: 10.00-17.00 (каждый день). Место: г. Алматы, ул. Майлина, 230. "
    "Стоимость: 100 000 ТГ. Запись по предоплате от 35000 тг. "
    "Планируете записаться на семинар?\n"
    "Клиент: К большому сожалению, я не смогу принять участие в курсе."
)

# Кусок кода на Python (комментарии на русском). Код рабочий: считает
# предоплату и печатает цены в тенге.
PYTHON_CODE = '''\
def prepayment(price_tg: int, percent: float = 0.35) -> int:
    # Предоплата за семинар: по умолчанию 35% от стоимости, в тенге.
    return round(price_tg * percent)


def format_tg(amount: int) -> str:
    # 100000 -> "100 000 тг"
    return f"{amount:,}".replace(",", " ") + " тг"


seminars = {"Расширенный курс": 100_000, "Мастер-класс": 45_000}
for name, price in seminars.items():
    print(f"{name}: {format_tg(price)}, предоплата от {format_tg(prepayment(price))}")
'''.rstrip()

# Текст с числами, датами и суммами. Арифметика сверена:
# 29 + 18 = 47; 29 × 35 000 + 18 × 100 000 = 2 815 000; 29 × 65 000 = 1 885 000;
# 29 / 47 = 61,7%; 18 / 47 = 38,3%; 2 815 000 / 47 ≈ 59 893,62; 60 − 47 = 13.
NUMBERS_TEXT = (
    "Итоги регистрации на семинар за период с 01.09.2026 по 18.10.2026: "
    "зарегистрировано 47 участников, из них 29 (61,7%) внесли предоплату "
    "от 35 000 тг, 18 (38,3%) оплатили полную стоимость 100 000 тг. "
    "Общая сумма поступлений: 29 × 35 000 + 18 × 100 000 = 2 815 000 тг. "
    "Ожидаемый остаток к 18.10.2026: 29 × 65 000 = 1 885 000 тг. "
    "Средний чек: 59 893,62 тг. Свободных мест: 13 из 60."
)

REAL_TEXTS = {
    "Сообщение клиента": CLIENT_MESSAGE,
    "Переписка менеджер-клиент": CHAT_DIALOG,
    "Код на Python": PYTHON_CODE,
    "Числа, даты, суммы": NUMBERS_TEXT,
}

# Все тексты вместе — для подсчёта.
ALL_TEXTS = {**TEXTS, **REAL_TEXTS}


@dataclass
class Tokenizer:
    """Результат подсчёта токенов одним токенизатором для всех текстов."""

    label: str  # полное название — для подробной таблицы
    short: str  # короткое название — для колонки сводной таблицы
    counts: dict[str, int]  # название текста -> число токенов (пусто = н/д)
    note: str = field(default="")  # пояснение под заголовком таблицы


def tiktoken_tokenizers() -> list[Tokenizer]:
    """Считает токены локально через tiktoken."""
    result = []
    for enc_name in TIKTOKEN_ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        counts = {}
        for name, text in ALL_TEXTS.items():
            tokens = enc.encode(text)
            assert enc.decode(tokens) == text  # проверка обратимости
            counts[name] = len(tokens)
        result.append(
            Tokenizer(f"tiktoken {enc_name}", enc_name.removesuffix("_base"), counts)
        )
    return result


def count_with_claude(
    client: anthropic.Anthropic, model: str
) -> tuple[dict[str, int], int]:
    """Считает токены текстов через count_tokens, вычитая служебную добавку."""

    def raw(text: str) -> int:
        return client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        ).input_tokens

    overhead = raw(OVERHEAD_PROBE) - 1
    return {name: raw(text) - overhead for name, text in ALL_TEXTS.items()}, overhead


def claude_tokenizers() -> list[Tokenizer]:
    """Считает токены Claude. Без ключа или при ошибке не падает, а поясняет."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "\nКлюч ANTHROPIC_API_KEY не найден — токены Claude пропущены.\n"
            "Скопируйте .env.example в .env и вставьте ключ "
            "(файл .env не попадает в git)."
        )
        return []

    print("\nСчитаю токены Claude через count_tokens...")
    client = anthropic.Anthropic()  # ключ берётся из ANTHROPIC_API_KEY
    tokenizers: list[Tokenizer] = []
    for model in CLAUDE_MODELS:
        short = model.removeprefix("claude-")
        try:
            counts, overhead = count_with_claude(client, model)
        except anthropic.AuthenticationError:
            print("  Ключ отклонён API — проверьте ANTHROPIC_API_KEY в файле .env.")
            return []
        except (anthropic.PermissionDeniedError, anthropic.NotFoundError):
            print(f"  {model}: нет доступа к модели или её не существует — пропускаю.")
        except anthropic.RateLimitError:
            print(f"  {model}: превышен лимит запросов — пропускаю, повторите позже.")
        except anthropic.APIStatusError as e:
            print(f"  {model}: ошибка API {e.status_code} — пропускаю.")
        except anthropic.APIConnectionError:
            print("  Нет связи с API — остальные модели пропускаю.")
            tokenizers.extend(
                Tokenizer(f"Claude {m}", m.removeprefix("claude-"), {})
                for m in CLAUDE_MODELS[len(tokenizers):]
            )
            break
        else:
            print(f"  {model}: готово (служебная добавка запроса: {overhead} ток.)")
            tokenizers.append(
                Tokenizer(
                    f"Claude {model} (count_tokens)",
                    short,
                    counts,
                    f"из каждого числа вычтена служебная добавка запроса: {overhead} ток.",
                )
            )
            continue
        tokenizers.append(Tokenizer(f"Claude {model}", short, {}))
    return tokenizers


def print_detail(tok: Tokenizer) -> None:
    """Подробные таблицы одного токенизатора: для обеих групп текстов."""
    print(f"\n\n=== {tok.label} ===")
    if tok.note:
        print(f"({tok.note})")

    # Группа 1: языки, с отношением к английскому.
    names = list(TEXTS)
    base_tokens = tok.counts[names[0]]  # английский — точка отсчёта
    header = (
        f"{'Один текст на разных языках':<{NAME_W}}{'Симв.':>7}{'Байт':>7}{'Слов':>6}"
        f"{'Токенов':>9}{'Симв./ток.':>12}{'Ток./слово':>12}{'× к англ.':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, text in TEXTS.items():
        n_tokens = tok.counts[name]
        print(
            f"{name:<{NAME_W}}{len(text):>7}{len(text.encode('utf-8')):>7}"
            f"{len(text.split()):>6}{n_tokens:>9}{len(text) / n_tokens:>12.2f}"
            f"{n_tokens / len(text.split()):>12.2f}{n_tokens / base_tokens:>11.2f}"
        )

    # Группа 2: реальные тексты, без отношения к английскому (перевода нет).
    print()
    header = (
        f"{'Реальные тексты (русский)':<{NAME_W}}{'Симв.':>7}{'Байт':>7}{'Слов':>6}"
        f"{'Токенов':>9}{'Симв./ток.':>12}{'Ток./слово':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, text in REAL_TEXTS.items():
        n_tokens = tok.counts[name]
        print(
            f"{name:<{NAME_W}}{len(text):>7}{len(text.encode('utf-8')):>7}"
            f"{len(text.split()):>6}{n_tokens:>9}{len(text) / n_tokens:>12.2f}"
            f"{n_tokens / len(text.split()):>12.2f}"
        )


def print_summary(tokenizers: list[Tokenizer]) -> None:
    """Сводные таблицы: все токенизаторы рядом."""
    width = 12

    def table(title: str, names: list[str], cell) -> None:
        print(f"\n\n=== {title} ===")
        header = f"{'Текст':<{NAME_W}}" + "".join(
            f"{t.short:>{width}}" for t in tokenizers
        )
        print(header)
        print("-" * len(header))
        for name in names:
            row = f"{name:<{NAME_W}}"
            for tok in tokenizers:
                row += f"{cell(tok, name):>{width}}"
            print(row)

    def tokens_cell(tok: Tokenizer, name: str) -> str:
        return str(tok.counts[name]) if tok.counts else "н/д"

    def ratio_cell(tok: Tokenizer, name: str) -> str:
        if not tok.counts:
            return "н/д"
        return f"{tok.counts[name] / tok.counts[next(iter(TEXTS))]:.2f}"

    def chars_cell(tok: Tokenizer, name: str) -> str:
        if not tok.counts:
            return "н/д"
        return f"{len(ALL_TEXTS[name]) / tok.counts[name]:.2f}"

    table("СВОДКА: число токенов", list(ALL_TEXTS), tokens_cell)
    table(
        "СВОДКА: во сколько раз больше токенов, чем в английском (группа 1)",
        list(TEXTS),
        ratio_cell,
    )
    table("СВОДКА: символов на токен (больше = экономнее)", list(ALL_TEXTS), chars_cell)


def main() -> None:
    # Чтобы кириллица и диакритика печатались и в консоли Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    # Читаем ключ из файла .env; значение из .env важнее системных переменных.
    load_dotenv(override=True)

    print("Тексты\n" + "=" * 6)
    for name, text in ALL_TEXTS.items():
        print(f"\n[{name}]\n{text}")

    tokenizers = tiktoken_tokenizers() + claude_tokenizers()

    for tok in tokenizers:
        if tok.counts:
            print_detail(tok)
    print_summary(tokenizers)


if __name__ == "__main__":
    main()
