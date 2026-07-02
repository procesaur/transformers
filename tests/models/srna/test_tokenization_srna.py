# Copyright 2026 Mihailo Škorić. All rights reserved.
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


import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers.models.srna.tokenization_srna import SrnaTokenizer
from transformers.testing_utils import (
    require_tokenizers,
)

@require_tokenizers
class SrnaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "procesaur/Srna_tokenizer"
    tokenizer_class = SrnaTokenizer

    integration_expected_tokens = ['<capi>', 't', 'his', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁ', 'ĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġbor', 'n', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfal', 's', 'Ã©', '.', 'Ċ', 'ç', 'Ķ', 'Ł', 'æ', '´', '»', 'ç', 'ļ', 'Ħ', 'ç', 'ľ', 'Ł', 'è', '°', 'Ľ', 'æ', 'ĺ', '¯', '<capi>', 'Ċ', 'hi', 'Ġ', '<capi>', 'Ġhe', 'llo', '<capi>', 'Ċ', 'hi', 'Ġ', 'Ġ', '<capi>', 'Ġhe', 'llo', 'Ċ', 'Ċ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', '<capi>', 'Ġhe', 'llo', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'the', 're', '<capi>', 'Ċ', 'the', 'Ġfol', 'lo', 'wing', 'Ġstri', 'ng', 'Ġsho', 'uld', 'Ġbe', 'Ġpro', 'per', 'ly', 'Ġen', 'code', 'd', ':', '<capi>', 'Ġhe', 'llo', '.', '<capi>', 'Ċ', 'bu', 't', 'Ġi', 'rd', 'Ġand', 'Ġ', 'à', '¸', 'Ľ', 'à', '¸', 'µ', 'Ġ', 'Ġ', 'Ġi', 'rd', 'Ġ', 'Ġ', 'Ġ', 'à', '¸', 'Ķ', '<capi>', 'Ċ', 'he', 'y', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdo', 'ing']  # fmt: skip
    integration_expected_token_ids = [2, 88, 9501, 476, 314, 4147, 33963, 251, 237, 203, 45, 13359, 2016, 82, 400, 984, 8642, 16, 1233, 13504, 476, 12947, 87, 2346, 18, 203, 168, 247, 258, 167, 117, 124, 168, 253, 231, 168, 255, 258, 169, 113, 254, 167, 251, 112, 2, 203, 636, 225, 2, 1237, 14316, 2, 203, 636, 225, 225, 2, 1237, 14316, 203, 203, 225, 203, 225, 225, 203, 2, 1237, 14316, 203, 32, 87, 34, 203, 636, 32, 87, 34, 7190, 271, 2, 203, 7190, 8896, 305, 28065, 6195, 699, 13153, 32130, 546, 335, 1688, 3002, 1916, 11981, 72, 30, 2, 1237, 14316, 18, 2, 203, 595, 88, 265, 4005, 1233, 225, 161, 121, 254, 161, 121, 118, 225, 225, 265, 4005, 225, 225, 225, 161, 121, 247, 2, 203, 608, 93, 39443, 10062, 13111, 325, 11456]  # fmt: skip
    expected_tokens_from_ids = ['<capi>', 't', 'his', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁ', 'ĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġbor', 'n', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfal', 's', 'Ã©', '.', 'Ċ', 'ç', 'Ķ', 'Ł', 'æ', '´', '»', 'ç', 'ļ', 'Ħ', 'ç', 'ľ', 'Ł', 'è', '°', 'Ľ', 'æ', 'ĺ', '¯', '<capi>', 'Ċ', 'hi', 'Ġ', '<capi>', 'Ġhe', 'llo', '<capi>', 'Ċ', 'hi', 'Ġ', 'Ġ', '<capi>', 'Ġhe', 'llo', 'Ċ', 'Ċ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', '<capi>', 'Ġhe', 'llo', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'the', 're', '<capi>', 'Ċ', 'the', 'Ġfol', 'lo', 'wing', 'Ġstri', 'ng', 'Ġsho', 'uld', 'Ġbe', 'Ġpro', 'per', 'ly', 'Ġen', 'code', 'd', ':', '<capi>', 'Ġhe', 'llo', '.', '<capi>', 'Ċ', 'bu', 't', 'Ġi', 'rd', 'Ġand', 'Ġ', 'à', '¸', 'Ľ', 'à', '¸', 'µ', 'Ġ', 'Ġ', 'Ġi', 'rd', 'Ġ', 'Ġ', 'Ġ', 'à', '¸', 'Ķ', '<capi>', 'Ċ', 'he', 'y', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdo', 'ing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"
