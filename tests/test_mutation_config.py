"""Mutation tests of `rizzo_flow.config`."""

import pytest
from test_mutation_support import use_ascii_as_default_encoding

from rizzo_flow import config


@pytest.fixture
def ascii_default(monkeypatch, tmp_path_factory):
    use_ascii_as_default_encoding(monkeypatch, tmp_path_factory.mktemp("probe"))


@pytest.fixture
def no_token_anywhere(tmp_path, monkeypatch):
    """Neither variable, no dotenv file in the working directory, no token saved by `hf`."""
    monkeypatch.chdir(tmp_path)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    return tmp_path


def test_the_dotenv_file_is_read_as_utf8_whatever_the_platform_default(
    no_token_anywhere, ascii_default
):
    """`hf_token` reading the dotenv file without `encoding="utf-8-sig"`, or with `encoding=None`.

    Bytes that are not valid in the encoding are replaced, not an error, so the value has to hold
    one: with ASCII for a default the token would come back with two replacement characters."""
    dotenv = no_token_anywhere / ".env"
    token = "hf_d\N{LATIN SMALL LETTER O WITH DIAERESIS}tenv"
    dotenv.write_bytes(f"# le chiavi \N{CHECK MARK} di qui\nHF_TOKEN={token}\n".encode())
    assert config.hf_token() == token


def test_the_token_saved_by_the_hub_client_is_read_as_utf8_whatever_the_platform_default(
    no_token_anywhere, ascii_default
):
    """`hf_token` reading `$HF_HOME/token` without `encoding="utf-8"`, or with `encoding=None`."""
    home = no_token_anywhere / "hf-home"
    home.mkdir()
    (home / "token").write_bytes("hf_t\N{LATIN SMALL LETTER O WITH DIAERESIS}ken\n".encode())
    assert config.hf_token() == "hf_t\N{LATIN SMALL LETTER O WITH DIAERESIS}ken"


def test_a_dotenv_token_of_nothing_but_quotes_and_blanks_is_no_token(no_token_anywhere):
    """`hf_token` testing the value without stripping its blanks first: `HF_TOKEN= "" ` would
    pass the test, and the empty string would be returned as the token."""
    dotenv = no_token_anywhere / ".env"
    dotenv.write_text('HF_TOKEN= "" \n', encoding="utf-8")
    assert config.hf_token() is None


@pytest.mark.parametrize(
    ("line", "value"),
    [
        ("A=x  # note", "x"),  # the comment starts at the last blank before `#`, not at the first
        ("A=x \t # note", "x"),
        ("A=x y   # note", "x y"),  # blanks inside the value stay, the ones before `#` do not
        ("A=   x   # note", "x"),  # and the ones in front of the value were never part of it
    ],
)
def test_the_blanks_before_a_trailing_comment_are_not_part_of_the_value(line, value):
    """`parse_dotenv` trimming the start of the text before the comment instead of its end
    (`.lstrip()` for `.rstrip()`): only the blank right in front of `#` goes with the comment, so
    a value followed by more than one kept the others, and `x # note` alone cannot tell."""
    assert config.parse_dotenv(line) == [("A", value)]
