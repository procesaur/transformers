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

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging

import regex

logger = logging.get_logger(__name__)



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

reverse_mapping = {v: k for k, v in mapping.items()}
reverse_mapping["dj"] = "ђ"

digraphExceptions = {
    "dj": ["adjektiv", "adjon", "adjunkt", "adjuvat", "adjudik", "adjung", "adjunk", "adjutant", "autodjel", "bazdje", "bdje",
    "bezdje", "blijedje", "bludje", "bordje", "bridjе", "burdje", "vidjel", "vidjet", "vindjakn", "višedjel", "višenedje",
    "vrijedje", "vododjelni", "vodosnabdje", "gdje", "gudje", "gdjir", "daladje", "daždje", "dvodjel", "dvonedje", "devetonedje",
    "depardje", "desetonedje", "didje", "djb", "djeva", "djevi", "djevo", "djevstv", "djed", "djejstv", "djel", "djenem", "djeneš",
    "djenu", "djet", "djec", "dječ", "djuar", "djubison", "djubouz", "djuer", "djui", "djuks", "djulej", "djumars", "djupont",
    "djurant", "djusenberi", "djuharst", "djuherst", "dovdje", "dogrdje", "dodjel", "dosmrdj", "dugodjelujuć", "drvodje",
    "drugdje", "elektrosnabdje", "endjurans", "žudje", "zabludje", "zavidje", "zavrijedje", "zagudje", "zadjev", "zadjen",
    "zalebdje", "zaludje", "zaodje", "zapodje", "zaprdj", "zarudje", "zasjedje", "zasmrdje", "zastidje", "zaštedje", "zdje",
    "zemljedjel", "zlodje", "igdje", "izbledje", "izblijedje", "izvidje", "izdjejst", "izdjelj", "izludje", "isprdje", 
    "jednodjel", "jednonedje", "koadjutor", "kojegdje", "kudjelj", "lebdje", "ludjel", "ludjet", "makfadjen", "marmadjuk",
    "međudjel", "najdjelatn", "najdjelotvornij", "nadjaha", "nadjača", "nadjeb", "nadjev", "nadježd", "nadjezik", "nadjezič", 
    "nadjenul", "nadjenuo", "nadjenut", "nasmrdj", "neadjuvantne", "negdje", "nedjel", "nadjunač", "nenadjača", "nenadjebi",
    "nenavidje", "neoadjuvant", "neodje", "nepodjarm", "neopredjel", "nepodjeljenoj", "nerazdje", "nigdje", "obdjel", 
    "obnevidje", "ovdje", "odjav", "odjad", "odjah", "odjaš", "odjeb", "odjev", "odjed", "odjezd", "odjek", "odjel", 
    "odjen", "odjeć", "odjec", "odjur", "odsjedje", "odjesen", "odječ", "ondje", "opredje", "oskudje", "osijedje",
    "osmonedje", "otprdjeti", "pardju", "perdju", "petodjel", "petonedje", "poblijedje", "povidje", "pogdjegdje", "pogdje", 
    "podjaz", "podjakn", "podjamč", "podjastrebačk", "podjastu", "podjemč", "podjar", "podjeb", "podjed", "podjezer",
    "podjezik", "podjezic", "podjezič", "podjel", "podjen", "podjesen", "podjet", "podjužnoslovenski", "podjurisdikcije", 
    "podjurne", "podjurnu", "podjuhorskom", "pododjel", "podrazdjel", "pozavidje", "poludje", "poljodjel", "ponegdje",
    "ponedjelj", "poodjača", "poodjezdi", "poodjeknu", "poodjutri", "popridjevljen", "porazdje", "posijedje", "posjedje",
    "postadjektiv", "postidje", "potpodjel", "poštedje", "pradjed", "prdje", "preblijedje", "previdje", "predvidje",
    "preadjektiv", "predjel", "preodjen", "preraspodje", "presjedje", "pridjev", "pridjen", "predjugosl", "predjulijansk",
    "predjustinijansk", "predjutar", "predjutr", "presmrdje", "prisnodjev", "prismrdje", "prištedje", "probdje",
    "problijedje", "prodjen", "prolebdje", "prosijedje", "prosjedje", "prosmrdje", "protivdjel", "prošlonedje", "radjard", 
    "razvidje", "razdjev", "razdjel", "razodje", "raspodje", "rasprdje", "remekdjel", "rudjen", "rudjet", "samoadjungovan",
    "samodjel", "samoopredjelje", "sadje", "svagdadjev", "svagdje", "svidje", "svugdje", "sedmonedjelj", "sijedje",
    "sjedje", "smrdje", "snabdje", "snovidje", "sredjužnoslovensk", "starosjedje", "stidje", "studje", "sudjel", "transdjus",
    "trejdjunion", "trodjeln", "tronedje", "ublijedje", "uvidje", "udjel", "udjen", "uprdje", "usidjel", "usjedje", "usmrdje", 
    "uštedje", "cjelonedje", "četverodjel", "četvoronedje", "čukundjed", "šestonedjelj", "štedje", "štogdje", "šukundjed"],

    "dž": ["feldžandarm", "nadžanj", "nadžanr", "nadždrel", "nadžel", "nadžeo", "nadžet", "nadživ", "nadžinj", "nadžnj",
    "nadžrec", "nadžup", "odžali", "odžari", "odžel", "odživ", "odžubor", "odžvaka", "odžval", "odžvać", "podžanr", 
    "podžargon", "podžel", "podže", "podžig", "podžiz", "podžil", "podžnje", "podžupan", "predželu", "predžetven", "predživot"],

    "nj": ["anjon", "benjamin", "vanjezičk", "vanjezičn", "guanju", "dianjon", "injaric", "injekc", "injekt", "injicira",
    "injurecesij", "injurij", "izvanjezičk", "izopolianjon", "kenjon", "konjug", "konjunk", "monoanjon", "nekonjug", 
    "nekonjunk", "netanjahu", "oksianjon", "panjevrej", "panjelinsk", "panjeremen", "panjugosl", "panjudej", "panjuridizacij", 
    "pinjin", "polianjon", "ssrnj", "subkonjunkt", "tanjug", "čanjol", "šenjang"],
    
    "lj": ["dablju", "epljard", "hiljemark", "metiljodid", "teljurajd"]
    }


def cyr2lat(text: str) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def lat2cyr(text: str) -> str:
    result = []
    word_buffer = ""
    i = 0
    while i < len(text):
        ch = text[i]

        # If character not in any mapping key (single or digraph start), flush buffer
        if not any(ch.lower() == k[0] for k in reverse_mapping.keys()):
            if word_buffer:
                result.append(_convert_word(word_buffer))
                word_buffer = ""
            result.append(ch)  # keep the non‑mapped char (space, number, punctuation)
            i += 1
            continue

        # Otherwise accumulate into buffer
        word_buffer += ch
        i += 1

    # Flush last buffer
    if word_buffer:
        result.append(_convert_word(word_buffer))

    return "".join(result)

def _convert_word(word: str) -> str:
    i = 0
    out = []
    word_lower = word.lower()  # compute once
    while i < len(word):
        matched = False
        for digraph in ["dž", "lj", "nj", "dj"]:
            if word_lower[i:i+len(digraph)] == digraph:
                # check exceptions for this digraph
                if any(word_lower.startswith(exc) for exc in digraphExceptions[digraph]):
                    # expand into individual letters
                    for ch in digraph:
                        out.append(reverse_mapping.get(ch, ch))
                    i += len(digraph)
                else:
                    out.append(reverse_mapping[word[i:i+len(digraph)]])
                    i += len(digraph)
                matched = True
                break
        if not matched:
            out.append(reverse_mapping.get(word[i], word[i]))
            i += 1
    return "".join(out)


class SrnaTokenizer(TokenizersBackend):
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        vocab_file=None,
        merges_file=None,
        unk_token: str = "<|endoftext|>",
        bos_token=None,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        add_prefix_space=None,
        
        boc_token = "<|cyr_start|>",
        eoc_token = "<|cyr_end|>",
        cap_token = "<|cap|>",
        up_token = "<|up|>",
        case_compression = True,
        script_compression = True,
        omit_tags = False,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space if add_prefix_space is not None else False
        self._vocab = (
            vocab
            if vocab is not None
            else {
                "<|endoftext|>": 0,
            }
        )

        self.boc_token = boc_token
        self.eoc_token = eoc_token
        self.cap_token = cap_token
        self.up_token = up_token
        self.case_compression = case_compression
        self.script_compression = script_compression
        self.omit_tags = omit_tags


        self.PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self.CYRILLIC_REGEX = r"\s?\p{Cyrillic}[^\p{Latin}]*"
        self.CLEANUP_FIND = rf"(\s+){regex.escape(self.eoc_token)}"
        self.CLEANUP_REPLACE = rf"{regex.escape(self.eoc_token)}\1"
        self.CAPITAL_REGEX = r"\s?\b\p{Lu}\p{Ll}+\b"
        self.UPPER_REGEX = r"\s?\b\p{Lu}{2,}\b"
        self.UP_FIND = rf"{regex.escape(self.up_token)}(\s?\w+)"
        self.CAP_FIND = rf"{regex.escape(self.cap_token)}(\s?\w+)"
        self.CYR_FIND = rf"{regex.escape(self.boc_token)}(.*?)({regex.escape(self.eoc_token)}|$)"

        self._merges = merges or []
        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.normalizer = normalizers.NFC()
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(self.PRETOKENIZE_REGEX),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space,
                    use_regex=False,
                ),
            ]
        )

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
    
    def prepare_for_tokenization(self, text: str, is_split_into_words=False, **kwargs):
        def mark_cap(match):
            if self.omit_tags:
                return match.group(0).lower()
            return f"{self.cap_token}{match.group(0).lower()}"

        def mark_up(match):
            if self.omit_tags:
                return match.group(0).lower()
            return f"{self.up_token}{match.group(0).lower()}"

        def wrap_and_transliterate(match):
            cyr_text = match.group(0)
            latin_text = cyr2lat(cyr_text)
            if self.omit_tags:
                return latin_text
            return f"{self.boc_token}{latin_text}{self.eoc_token}"

        if self.script_compression:
            text = regex.sub(self.CYRILLIC_REGEX, wrap_and_transliterate, text)
            if text.endswith(self.eoc_token):
                text = text[: -len(self.eoc_token)]
            text = regex.sub(self.CLEANUP_FIND, self.CLEANUP_REPLACE, text)
        if self.case_compression:
            text = regex.sub(self.CAPITAL_REGEX, mark_cap, text)
            text = regex.sub(self.UPPER_REGEX, mark_up, text)
        return text

    def encode(self, text: str, *args, **kwargs):
        text = self.prepare_for_tokenization(text)
        return super().encode(text, *args, **kwargs)

    def _encode_plus(self, text, *args, **kwargs):
        if self.add_prefix_space:
            text = text.lstrip("\n\t")
            if not text.startswith(" "):
                text = " " + text
        # apply your preprocessing hook
        text = self.prepare_for_tokenization(text, **kwargs)
        return super()._encode_plus(text, *args, **kwargs)


    def decode(self, token_ids, *args, **kwargs):
        text = super().decode(token_ids, *args, **kwargs)
        # then your custom tag decoding

        def decode_cyr(match):
            return lat2cyr(match.group(1))

        def decode_cap(match):
            inner = match.group(1)
            leading_ws = regex.match(r"^\s*", inner).group(0)
            if not leading_ws:
                return inner[0].upper() + inner[1:]

            core = inner.lstrip()
            if core:
                core = core[0].upper() + core[1:]
            return leading_ws + core

        def decode_allcaps(match):
            inner = match.group(1)
            return inner.upper()
   
        if self.case_compression:
            text = regex.sub(self.UP_FIND, decode_allcaps, text)
            text = regex.sub(self.CAP_FIND, decode_cap, text)
        if self.script_compression:
            text = regex.sub(self.CYR_FIND, decode_cyr, text)
        return text


__all__ = ["SrnaTokenizer"]
