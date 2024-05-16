import re
from typing import Optional
from num2words import num2words
from googletrans import Translator
from eld import LanguageDetector


def number_to_string(num: int, lang: str):
    translator = Translator()
    text_en = num2words(num)
    text = translator.translate(text_en, src="en", dest=lang).text
    return text


def standardize(text: str, ensure_lang: Optional[str] = None) -> Optional[str]:
    detector = LanguageDetector()
    detected_lang = detector.detect(text).language
    ensure_lang = ensure_lang or detected_lang
    if ensure_lang != detected_lang:
        return None

    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    def replace_numbers(match):
        num = int(match.group())
        return number_to_string(num, ensure_lang)

    text = re.sub(r"\d+", replace_numbers, text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;?!])", r"\1", text)

    return text


if __name__ == "__main__":
    text = "Nějaký    divý kus textu (remove this), [same here] 2 + 2 = 4."
    standardized = standardize(text, ensure_lang="cs")
    print(standardized)
