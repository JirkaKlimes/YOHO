import re
from typing import Optional
from num2words import num2words


def standardize_text(text: str, lang: str) -> Optional[str]:
    # num2words doesn't use ISO-639 codes
    corrections = {
        "cs": "cz",
        "da": "dk",
        "tgk": "tg",
    }
    if lang in corrections:
        lang = corrections[lang]

    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    def replace_numbers(match):
        num = match.group()
        num = num.replace(",", ".")
        text = num2words(num, lang=lang)
        return text

    text = re.sub(r"\d+([.,]\d+)?", replace_numbers, text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;?!])", r"\1", text)

    return text


if __name__ == "__main__":
    text = "Nějaký 3.3   divý kus 124,512 textu (remove this), [same here] 2 + 2 = 4."
    standardized = standardize_text(text, lang="cs")
    print(standardized)
