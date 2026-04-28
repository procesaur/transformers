# Copyright 2026 Mihailo Škorić.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for Serbian and Serbo-Croatian models."""

from tokenizers import decoders, normalizers
#from ...utils import logging
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

import regex

#logger = logging.get_logger(__name__)

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
CYRILLIC_REGEX = r"\s?\p{Cyrillic}[^\p{Latin}]*"
CLEANUP_FIND = r"(\s+)</cyr>"
CLEANUP_REPLACE = r"</cyr>\1"
CAPITAL_REGEX = r"\s?\b\p{Lu}\p{Ll}+\b"
UPPER_REGEX = r"\s?\b\p{Lu}{2,}\b"
cyr_open = "<cyr>"
cyr_close = "</cyr>"
cap_tag = "<cap>"
up_tag = "<up>"
UP_FIND = r"<up>(\s?\w+)"
CAP_FIND = r"<cap>(\s?\w+)"
CYR_FIND = r"<cyr>(.*?)</cyr>"

mapping = {
        "Љ": "Lj", "љ": "lj", "Њ": "Nj", "њ": "nj", "Џ": "Dž", "џ": "dž",
        "А": "A", "а": "a", "Б": "B", "б": "b", "В": "V", "в": "v",
        "Г": "G", "г": "g", "Д": "D", "д": "d", "Ђ": "Đ", "ђ": "đ",
        "Е": "E", "е": "e", "Ж": "Ž", "ж": "ž", "З": "Z", "з": "z",
        "И": "I", "и": "i", "Ј": "J", "ј": "j", "К": "K", "к": "k",
        "Л": "L", "л": "l", "М": "M", "м": "m", "Н": "N", "н": "n",
        "О": "O", "о": "o", "П": "P", "п": "p", "Р": "R", "р": "r",
        "С": "S", "с": "s", "Т": "T", "т": "t", "Ћ": "Ć", "ћ": "ć",
        "У": "U", "у": "u", "Ф": "F", "ф": "f", "Х": "H", "х": "h",
        "Ц": "C", "ц": "c", "Ч": "Č", "ч": "č", "Ш": "Š", "ш": "š",
    }

def cyr2lat(text: str) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def lat2cyr(text: str) -> str:
    return "".join({v: k for k, v in mapping.items()}.get(ch, ch) for ch in text)


def SrnaNormalize(text):
    def mark_cap(match):
            return f"{cap_tag}{match.group(0).lower()}"

    def mark_up(match):
            return f"{up_tag}{match.group(0).lower()}"

    def wrap_and_transliterate(match):
            cyr_text = match.group(0)
            latin_text = cyr2lat(cyr_text)
            return f"{cyr_open}{latin_text}{cyr_close}"

    text = regex.sub(CYRILLIC_REGEX, wrap_and_transliterate, text)
    text = regex.sub(CLEANUP_FIND, CLEANUP_REPLACE, text)
    text = regex.sub(CAPITAL_REGEX, mark_cap, text)
    return regex.sub(UPPER_REGEX, mark_up, text)


def SrnaDecode(text):
    # Decode <cyr>...</cyr>
    def decode_cyr(match):
        return lat2cyr(match.group(1))

    # Decode <cap>word
    def decode_cap(match):
        inner = match.group(1)
        # Preserve leading/trailing whitespace
        leading = len(inner) - len(inner.lstrip())
        trailing = len(inner) - len(inner.rstrip())

        core = inner.strip()
        if core:
            core = core[0].upper() + core[1:]  # manual capitalize
        # Reattach whitespace
        return (" " * leading) + core + (" " * trailing)

    # Decode <allcaps>word
    def decode_allcaps(match):
        inner = match.group(1)
        return inner.upper()

    text = regex.sub(UP_FIND, decode_allcaps, text)
    text = regex.sub(CAP_FIND, decode_cap, text)
    text = regex.sub(CYR_FIND, decode_cyr, text)
    return text


class SrnaTokenizer(Qwen2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def encode(self, text: str, *args, **kwargs):
        # Step 1: run the built-in normalizer
        text = super().normalize(text) if hasattr(super(), "normalize") else text        # Step 2: run your custom normalization
        normalized = SrnaNormalize(text)
        # Step 3: pass to the parent encode
        return super().encode(normalized, *args, **kwargs)

    def decode(self, token_ids, *args, **kwargs):
        text = super().decode(token_ids, *args, **kwargs)
        # then your custom tag decoding
        return SrnaDecode(text)

__all__ = ["SrnaTokenizer"]



tok = SrnaTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

original = "1. Hej, жабо, Поздрављам те, dobri svete, iz године 2029-те. Hadži-Jordan. POZZ!"
expected = "1.<cap> hej,<cyr> žabo,<cap> pozdravljam te,</cyr> dobri svete, iz<cyr> godine 2029-te.</cyr><cap> hadži-<cap>jordan.<up> pozz!"

# Example encoded text
encoded = "<cyr>Pozdrav</cyr> iz <cap>beograda <allcaps>srbijа"

# Use your custom decoder
ids = tok.encode(original)
print(ids)

decoded = tok.decode(ids)
print(decoded)