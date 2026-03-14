from __future__ import annotations

import unicodedata

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ASCII_DIGITS = "0123456789"

ARABIC_TO_PERSIAN_LETTER_MAP = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
    }
)

ARABIC_INDIC_TO_ASCII = str.maketrans({src: dst for src, dst in zip(ARABIC_INDIC_DIGITS, ASCII_DIGITS)})
ARABIC_INDIC_TO_PERSIAN = str.maketrans({src: dst for src, dst in zip(ARABIC_INDIC_DIGITS, PERSIAN_DIGITS)})
PERSIAN_TO_ASCII = str.maketrans({src: dst for src, dst in zip(PERSIAN_DIGITS, ASCII_DIGITS)})


def canonicalize_digits(text: str, target: str = "persian") -> str:
    if target == "persian":
        return text.translate(ARABIC_INDIC_TO_PERSIAN)
    if target == "ascii":
        return text.translate(ARABIC_INDIC_TO_ASCII).translate(PERSIAN_TO_ASCII)
    raise ValueError(f"Unsupported digit canonicalization target: {target}")


def canonicalize_equivalent_arabic_persian_letters(text: str, target: str = "persian") -> str:
    if target == "persian":
        return text.translate(ARABIC_TO_PERSIAN_LETTER_MAP)
    raise ValueError(f"Unsupported Arabic/Persian canonical letter target: {target}")


def canonicalize_arabic_persian_text(
    text: str,
    *,
    normalize_unicode: bool = False,
    digit_target: str = "persian",
    normalize_digits: bool = True,
    canonical_letter_target: str = "persian",
    normalize_equivalent_letters: bool = True,
) -> str:
    normalized = unicodedata.normalize("NFKC", text) if normalize_unicode else text
    if normalize_equivalent_letters:
        normalized = canonicalize_equivalent_arabic_persian_letters(
            normalized,
            target=canonical_letter_target,
        )
    if normalize_digits:
        normalized = canonicalize_digits(normalized, target=digit_target)
    return normalized
