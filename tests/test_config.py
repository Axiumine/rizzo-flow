"""Model and file registry, Hugging Face token discovery and the download paths.

Nothing touches the network and no weights are needed: `fetch` is replaced by a recorder, and the
token search runs in a temporary working directory with a temporary Hugging Face home.
"""

import dataclasses
import errno
import hashlib
import io
import itertools
from pathlib import Path

import pytest
from test_llama_release import Response

from rizzo_flow import config, llama_release


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A machine with no token anywhere: empty environment, working directory and hub home."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    return tmp_path


def write_dotenv(folder, text):
    # Bytes, so that Windows does not translate the line endings a test asks for.
    (folder / ".env").write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))


def save_login(folder, text):
    (folder / "hf").mkdir(exist_ok=True)
    (folder / "hf" / "token").write_text(text, encoding="utf-8")


@pytest.fixture
def fetched(monkeypatch):
    """Replaces `llama_release.fetch` and records what `download_gguf` asks it to do."""
    calls: list[dict] = []

    def fetch(url, target, sha256, progress=None, attempts=5, token=None):
        calls.append(
            {"url": url, "target": target, "sha256": sha256, "progress": progress, "token": token}
        )
        return target

    monkeypatch.setattr(llama_release, "fetch", fetch)
    return calls


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("prefill_chunk", [1, 512, 2048])
def test_the_limits_of_both_backends_include_their_bounds(batch_size, prefill_chunk):
    config.check_limits(batch_size, prefill_chunk)  # nothing to report


@pytest.mark.parametrize(
    ("batch_size", "prefill_chunk"),
    [(0, 512), (-1, 512), (17, 512), (4, 0), (4, -1), (4, 2049), (0, 0), (17, 2049)],
)
def test_the_limits_of_both_backends_refuse_what_is_outside_them(batch_size, prefill_chunk):
    with pytest.raises(ValueError, match=r"^batch_size must be 1–16 and prefill_chunk 1–2048$"):
        config.check_limits(batch_size, prefill_chunk)


# --- Hugging Face token ----------------------------------------------------------------------


def test_a_blank_token_variable_is_not_a_token(hub, monkeypatch):
    save_login(hub, "hf_login\n")
    monkeypatch.setenv("HF_TOKEN", "  \n")
    assert config.hf_token() == "hf_login"


@pytest.mark.parametrize("blank", [" ", "\n", " \t\r\n"])
def test_blank_variables_are_skipped_like_empty_ones_all_the_way_to_the_login(
    hub, monkeypatch, blank
):
    save_login(hub, "hf_login\n")
    monkeypatch.setenv("HF_TOKEN", blank)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", blank)
    assert config.hf_token() == "hf_login"
    write_dotenv(hub, "HF_TOKEN=from_dotenv\n")
    assert config.hf_token() == "from_dotenv"
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "  hf_alias\n")
    assert config.hf_token() == "hf_alias"  # a blank HF_TOKEN does not hide the alias


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HF_TOKEN=plain\n", "plain"),
        ('HF_TOKEN="double"\n', "double"),
        ("HF_TOKEN='single'\n", "single"),
        ("  HF_TOKEN  =  spaced  \n", "spaced"),
        ("HF_TOKEN=a=b==\n", "a=b=="),  # only the first equals sign splits
        ("# note\nOTHER=1\n\nHF_TOKEN=third\n", "third"),
        ("HF_TOKEN=first\nHF_TOKEN=second\n", "first"),
        ("OTHER=1\r\nHF_TOKEN=crlf\r\n", "crlf"),
        ("HF_TOKEN=no_final_newline", "no_final_newline"),
        ("HF_TOKEN=XXX\n", "XXX"),  # a token is opaque: no character of it is special
        ("HF_TOKEN=\nHF_TOKEN=second\n", "second"),  # an empty assignment does not end the search
        # What editors and shells add around the assignment.
        ("export HF_TOKEN=exported\n", "exported"),
        ("export  HF_TOKEN = spaced\n", "spaced"),
        ("HF_TOKEN=hf_abc # my token\n", "hf_abc"),
        ("HF_TOKEN=hf_abc  # my token\n", "hf_abc"),
        ("HF_TOKEN=hf_abc\t# my token\n", "hf_abc"),
        ('HF_TOKEN="hf_abc" # my token\n', "hf_abc"),
        ("HF_TOKEN='hf_abc' # \"my\" 'token'\n", "hf_abc"),
        ("HF_TOKEN=hf#abc\n", "hf#abc"),  # a # that no blank precedes is part of the value
        ("﻿HF_TOKEN=bom\n", "bom"),  # Notepad and PowerShell start a file with a BOM
        ("﻿# note\nHF_TOKEN=bom\n", "bom"),
        # Blanks inside the quotes are no token, and no part of one.
        ('HF_TOKEN="  padded  "\n', "padded"),
        ("HF_TOKEN=' \tpadded\t'\n", "padded"),
        ('HF_TOKEN=" "\nHF_TOKEN=second\n', "second"),
        # A quote that nothing closes is not part of the token either.
        ('HF_TOKEN="unclosed\n', "unclosed"),
        ("HF_TOKEN=unopened'\n", "unopened"),
    ],
)
def test_dotenv_assignments(hub, text, expected):
    write_dotenv(hub, text)
    assert config.hf_token() == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "OTHER=1\n",
        "HF_TOKEN\n",
        "HF_TOKEN=\n",
        "HF_TOKEN=''\n",
        'HF_TOKEN=""\n',
        "# HF_TOKEN=commented_out\n",
        "MY_HF_TOKEN=other\n",
        "HF_TOKEN_2=other\n",
        'HF_TOKEN=" "\n',
        "HF_TOKEN=' '\n",
        "HF_TOKEN=' \t'\n",
        'HF_TOKEN=" \'"\n',
        "export HF_TOKEN\n",
        "export HF_TOKEN=\n",
        "=hf_nameless\n",
    ],
)
def test_dotenv_without_a_usable_assignment_falls_through_to_the_login(hub, text):
    write_dotenv(hub, text)
    assert config.hf_token() is None
    save_login(hub, "hf_login\n")
    assert config.hf_token() == "hf_login"


def test_dotenv_token_survives_quotes_padding_comments_and_neighbours(hub):
    """The assignment in every dress an editor or a shell may give it, against every other."""
    neighbourhoods = [[], ["# note", "OTHER=1", "", "HF_TOKEN_X=y"]]
    dresses = itertools.product(
        ["hf_abc", "a=b/c+d.e-f_", "XXX"],  # token
        ["", "'", '"'],  # quote
        ["", " \t"],  # blanks between the quotes and the token
        ["", "  "],  # blanks around the assignment
        ["", " # note", "\t# note", "  #note"],  # trailing comment
        ["\n", "\r\n"],  # line end
        neighbourhoods,
    )
    for token, quote, inner, pad, comment, newline, neighbours in dresses:
        value = f"{quote}{inner}{token}{inner}{quote}"
        lines = [*neighbours, f"{pad}HF_TOKEN{pad}={pad}{value}{pad}{comment}"]
        write_dotenv(hub, newline.join(lines) + newline)
        assert config.hf_token() == token, lines


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("\n  \n\t\n", []),
        ("# only a note\n  # and another\n", []),
        ("# A=1\n  #B=2\nC=3\n", [("C", "3")]),  # a commented-out assignment is a note
        ("A=1\nB=2\n", [("A", "1"), ("B", "2")]),
        ("A=1\nA=2\n", [("A", "1"), ("A", "2")]),  # in order, none dropped: the caller decides
        ("  A  =  spaced  \n", [("A", "spaced")]),
        ("A=x=y==\n", [("A", "x=y==")]),
        ("A\nB=2\n", [("B", "2")]),  # no equals sign: not an assignment
        ("=1\n B=2\n", [("B", "2")]),  # no name
        ("export A=1\nexport  B = 2\n", [("A", "1"), ("B", "2")]),
        ("A=\nB=''\n", [("A", ""), ("B", "")]),
        ("A=x # note\nB=y\t# note\nC=z#w\n", [("A", "x"), ("B", "y"), ("C", "z#w")]),
        ("A='x # not a note' # a note\n", [("A", "x # not a note")]),
        ("A=\"x 'y' z\"\n", [("A", "x 'y' z")]),  # the other quote is text
        ("A='x'y'\n", [("A", "x")]),  # the first closing quote ends the value
        ('A="unclosed\n', [("A", '"unclosed')]),  # left as it is: the caller decides
        ("A=1\r\nB=2\r\n", [("A", "1"), ("B", "2")]),
        ("A=1\rB=2\n", [("A", "1"), ("B", "2")]),
    ],
)
def test_parse_dotenv_reads_assignments_in_order(text, expected):
    assert config.parse_dotenv(text) == expected


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
    assert config.parse_dotenv(line) == [("A", value)]


def test_a_dotenv_token_of_nothing_but_quotes_and_blanks_is_no_token(hub):
    write_dotenv(hub, 'HF_TOKEN= "" \n')
    assert config.hf_token() is None


@pytest.mark.parametrize(
    ("raw", "token"),
    [
        ("HF_TOKEN=hf_x\r\n".encode("utf-16"), None),  # what PowerShell 5.1 writes for `>`
        ("HF_TOKEN=hf_x\r\n".encode("utf-16-le"), None),  # the same, without the byte order mark
        ("HF_TOKEN=hf_x\r\n".encode("utf-16-be"), None),
        # A code page that is not UTF-8 costs the bytes that are not valid, and nothing else.
        ("# città\nHF_TOKEN=hf_x\n".encode("cp1252"), "hf_x"),
        ("HF_TOKEN=hf_x # perché\n".encode("latin-1"), "hf_x"),
        (b"\xff\xfe\x00\x80\nHF_TOKEN=hf_x\n", "hf_x"),
    ],
)
def test_a_dotenv_file_that_is_not_utf8_neither_stops_the_search_nor_hides_the_token(
    hub, raw, token
):
    write_dotenv(hub, raw)
    assert config.hf_token() == token
    save_login(hub, "hf_login\n")
    assert config.hf_token() == (token or "hf_login")  # the saved login is what follows


def test_the_dotenv_file_is_read_as_utf8_whatever_the_platform_default(hub, monkeypatch):
    """Windows reads a text file in its ANSI code page unless told otherwise; ASCII stands in for
    it here, on every platform. Bytes that are not valid are replaced and would show in the token."""

    def ascii_by_default(encoding, stacklevel=2):  # what `pathlib` asks for the default
        return "ascii" if encoding is None else encoding

    monkeypatch.setattr(io, "text_encoding", ascii_by_default)
    token = "hf_d\N{LATIN SMALL LETTER O WITH DIAERESIS}tenv"
    write_dotenv(hub, f"# le chiavi \N{CHECK MARK} di qui\nHF_TOKEN={token}\n")
    assert config.hf_token() == token


def test_a_dotenv_file_that_cannot_be_read_is_skipped_like_a_missing_one(hub, monkeypatch):
    write_dotenv(hub, "HF_TOKEN=hf_hidden\n")
    save_login(hub, "hf_login\n")
    read_text = Path.read_text

    def refuse_the_dotenv_file(self, *args, **options):
        if self.name == ".env":
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return read_text(self, *args, **options)

    monkeypatch.setattr(Path, "read_text", refuse_the_dotenv_file)
    assert config.hf_token() == "hf_login"
    (hub / "hf" / "token").unlink()
    assert config.hf_token() is None


def test_a_dotenv_file_in_another_encoding_does_not_stop_a_download(hub, fetched):
    write_dotenv(hub, "HF_TOKEN=hf_x\r\n".encode("utf-16"))
    assert config.download_gguf(variant="base") == config.gguf_spec(variant="base").path
    assert [call["token"] for call in fetched] == [None]


# --- downloads ---------------------------------------------------------------------------------


def test_a_destination_that_is_a_directory_gets_the_file_under_its_pinned_name(hub, fetched):
    (hub / "models").mkdir()
    spec = config.gguf_spec("1.7b", "bf16", "base")
    assert config.download_gguf("1.7b", "bf16", hub / "models", variant="base") == (
        hub / "models" / spec.file
    )
    assert [call["target"] for call in fetched] == [hub / "models" / spec.file]
    # A path that is not a directory is the file itself, whatever it is called or whether it exists.
    (hub / "old.gguf").write_bytes(b"an earlier download")
    for name in ("old.gguf", "new"):
        assert config.download_gguf(destination=hub / name) == hub / name
    assert [call["target"].name for call in fetched[1:]] == ["old.gguf", "new"]


def test_downloading_into_an_existing_directory_leaves_one_file_and_no_partial_one(
    hub, monkeypatch
):
    """The whole download used to land in `models.part` and fail on the rename onto a directory."""
    payload = b"GGUF stand-in for the weights"
    spec = dataclasses.replace(config.gguf_spec(), sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(config.GGUF, ("4b", "q8_0", "flow"), spec)
    requests = []

    def urlopen(request, timeout=None):
        requests.append(request.full_url)
        return Response(payload)

    monkeypatch.setattr(llama_release.urllib.request, "urlopen", urlopen)
    (hub / "models").mkdir()
    assert config.download_gguf(destination="models") == Path("models") / spec.file
    assert (hub / "models" / spec.file).read_bytes() == payload
    assert sorted(path.name for path in hub.iterdir()) == ["models"]  # no `models.part` beside it
    assert [path.name for path in (hub / "models").iterdir()] == [spec.file]
    assert requests == [spec.url]
