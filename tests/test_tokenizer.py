import pytest
from tokenizers import AddedToken
from transformers import AutoTokenizer

from balm.tokenizer import BalmTokenizer

SPECIAL = ["bos_token", "eos_token", "unk_token", "pad_token", "mask_token"]
DEFAULTS = {
    "bos_token": "<cls>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
}


@pytest.mark.parametrize("name", SPECIAL)
def test_special_token_given_as_added_token(name):
    content = DEFAULTS[name]
    as_str = BalmTokenizer()
    as_added = BalmTokenizer(**{name: AddedToken(content, special=True)})

    assert as_added("EVQLX")["input_ids"] == as_str("EVQLX")["input_ids"]
    assert str(getattr(as_added, name)) == content


def test_balm_moe_tokenizer_loads():
    tokenizer = AutoTokenizer.from_pretrained(
        "brineylab/BALM-MoE", revision="51842e69ca3f0ae3f239d5fbd854c78bc0857f82"
    )
    ids = tokenizer("EVQL<cls>DIQM")["input_ids"]

    assert tokenizer.convert_ids_to_tokens(ids) == [
        "<cls>", "E", "V", "Q", "L", "<cls>", "D", "I", "Q", "M", "<eos>"
    ]
