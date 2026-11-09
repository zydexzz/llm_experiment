"""Эксперимент: сколько токенов (tiktoken) занимает один и тот же текст
на английском, русском и казахском (кириллица и латиница).

Запуск:
    pip install -r requirements.txt
    python token_count.py
"""

from __future__ import annotations

import sys

import tiktoken

# Кодировки OpenAI: cl100k_base (GPT-4 / GPT-3.5), o200k_base (GPT-4o и новее).
ENCODINGS = ("cl100k_base", "o200k_base")

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


def main() -> None:
    # Чтобы кириллица и диакритика печатались и в консоли Windows.
    sys.stdout.reconfigure(encoding="utf-8")

    print("Тексты\n" + "=" * 6)
    for name, text in TEXTS.items():
        print(f"\n[{name}]\n{text}")

    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        rows = []
        for name, text in TEXTS.items():
            tokens = enc.encode(text)
            assert enc.decode(tokens) == text  # проверка обратимости
            rows.append((name, text, len(tokens)))

        base_tokens = rows[0][2]  # английский — точка отсчёта
        print(f"\n\n=== {enc_name} ===")
        header = (
            f"{'Текст':<24}{'Симв.':>7}{'Байт':>7}{'Слов':>6}"
            f"{'Токенов':>9}{'Симв./ток.':>12}{'Ток./слово':>12}{'× к англ.':>11}"
        )
        print(header)
        print("-" * len(header))
        for name, text, n_tokens in rows:
            chars = len(text)
            n_bytes = len(text.encode("utf-8"))
            words = len(text.split())
            print(
                f"{name:<24}{chars:>7}{n_bytes:>7}{words:>6}"
                f"{n_tokens:>9}{chars / n_tokens:>12.2f}{n_tokens / words:>12.2f}"
                f"{n_tokens / base_tokens:>11.2f}"
            )


if __name__ == "__main__":
    main()
