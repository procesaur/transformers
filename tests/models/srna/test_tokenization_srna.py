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
    from_pretrained_id = "procesaur/Srna_test"
    tokenizer_class = SrnaTokenizer

    integration_expected_tokens = ['<|cap|>', 'this', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁæ´»çļĦ', 'çľŁè°Ľ', 'æĺ¯', '<|cap|>', 'Ċ', 'hi', 'Ġ', '<|cap|>', 'Ġhello', '<|cap|>', 'Ċ', 'hi', 'ĠĠ', '<|cap|>', 'Ġhello', 'ĊĊ', 'ĠĊĠĠĊ', '<|cap|>', 'Ġhello', 'Ċ', '<s', '>', 'Ċ', 'hi', '<s', '>', 'there', '<|cap|>', 'Ċ', 'the', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', '<|cap|>', 'Ġhello', '.', '<|cap|>', 'Ċ', 'but', 'Ġ', 'ird', 'Ġand', 'Ġà¸Ľà¸µ', 'ĠĠ', 'Ġ', 'ird', 'ĠĠ', 'Ġà¸Ķ', '<|cap|>', 'Ċ', 'hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [248049, 559, 369, 264, 1228, 25677, 232, 198, 40, 557, 8950, 303, 220, 24, 17, 15, 15, 15, 11, 321, 411, 369, 30882, 933, 13, 198, 103815, 132339, 95761, 248049, 198, 5834, 220, 248049, 23066, 248049, 198, 5834, 256, 248049, 23066, 271, 46813, 248049, 23066, 198, 42589, 29, 198, 5834, 42589, 29, 17977, 248049, 198, 1719, 2614, 886, 1220, 381, 9971, 19873, 25, 248049, 23066, 13, 248049, 198, 7834, 220, 2517, 321, 170827, 256, 220, 2517, 256, 149027, 248049, 198, 34385, 1204, 513, 488, 3604]  # fmt: skip
    expected_tokens_from_ids = ['<|cap|>', 'this', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁæ´»çļĦ', 'çľŁè°Ľ', 'æĺ¯', '<|cap|>', 'Ċ', 'hi', 'Ġ', '<|cap|>', 'Ġhello', '<|cap|>', 'Ċ', 'hi', 'ĠĠ', '<|cap|>', 'Ġhello', 'ĊĊ', 'ĠĊĠĠĊ', '<|cap|>', 'Ġhello', 'Ċ', '<s', '>', 'Ċ', 'hi', '<s', '>', 'there', '<|cap|>', 'Ċ', 'the', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', '<|cap|>', 'Ġhello', '.', '<|cap|>', 'Ċ', 'but', 'Ġ', 'ird', 'Ġand', 'Ġà¸Ľà¸µ', 'ĠĠ', 'Ġ', 'ird', 'ĠĠ', 'Ġà¸Ķ', '<|cap|>', 'Ċ', 'hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"
