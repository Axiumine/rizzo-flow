"""Model and file registry, Hugging Face token discovery and the two download paths.

Nothing touches the network: `fetch` and `snapshot_download` are replaced by recorders (one test
scripts `urlopen` instead, to see what lands on disk), and the token search runs in a temporary
working directory with a temporary Hugging Face home.
"""

import dataclasses
import errno
import hashlib
import re
import string
import sys
import types
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_llama_release_extra import Response, script

from rizzo_flow import config, llama_release

PINNED_SIZES = {spec.hidden_size for spec in config.MODELS.values()}


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


# --- registry ------------------------------------------------------------------------------


def test_model_specs_are_pinned_and_paths_follow_the_repository_name():
    four, small = config.MODELS["4b"], config.MODELS["1.7b"]
    assert (four.repo, four.hidden_size) == ("XHToken/Spark-X2.5-4B", 2560)
    assert (small.repo, small.hidden_size) == ("XHToken/Spark-X2.5-1.7B", 2048)
    assert four.path == Path("models") / "Spark-X2.5-4B"
    assert small.path == Path("models") / "Spark-X2.5-1.7B"
    assert all(key == spec.size for key, spec in config.MODELS.items())
    assert all(re.fullmatch(r"[0-9a-f]{40}", spec.revision) for spec in config.MODELS.values())
    # The module-level shortcut follows the default size.
    assert config.DEFAULT_SIZE == "4b"
    assert four.repo == config.MODEL_ID


def test_defaults_are_the_fine_tune_at_q8_0():
    assert config.QUANTS == ("q8_0", "q4_k_m", "bf16")
    assert config.VARIANTS == ("flow", "base")
    assert (config.DEFAULT_QUANT, config.DEFAULT_VARIANT) == ("q8_0", "flow")
    assert config.gguf_spec() is config.GGUF[("4b", "q8_0", "flow")]


@pytest.mark.parametrize(("size", "hidden"), [("4b", 2560), ("1.7b", 2048)])
def test_identify_matches_on_the_hidden_size(size, hidden):
    found = config.identify({"hidden_size": hidden, "model_type": "spark2_5", "vocab_size": 1})
    assert found is config.MODELS[size]


def test_identify_refuses_unknown_checkpoints():
    unknown = (
        r"^Unrecognized Spark2\.5 checkpoint \(hidden_size=4096\); supported sizes: 4b, 1\.7b$"
    )
    with pytest.raises(ValueError, match=unknown):
        config.identify({"hidden_size": 4096})
    with pytest.raises(ValueError, match=r"\(hidden_size=None\)"):
        config.identify({"model_type": "spark2_5"})


@settings(max_examples=60, deadline=None, database=None, derandomize=True)
@given(hidden=st.integers().filter(lambda n: n not in PINNED_SIZES))
def test_identify_refuses_every_size_that_is_not_pinned(hidden):
    with pytest.raises(ValueError, match=rf"\(hidden_size={hidden}\)"):
        config.identify({"hidden_size": hidden})


def test_every_size_quant_and_variant_has_exactly_one_pinned_file():
    combinations = {
        (size, quant, variant)
        for size in config.MODELS
        for quant in config.QUANTS
        for variant in config.VARIANTS
    }
    assert set(config.GGUF) == combinations
    for (size, quant, variant), spec in config.GGUF.items():
        assert (spec.size, spec.quant, spec.variant) == (size, quant, variant)
        assert config.gguf_spec(size, quant, variant) is spec
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
        assert spec.file.endswith(".gguf")
        assert spec.path == Path("models") / spec.repo.split("/")[1] / spec.file
        # A commit, never a moving branch: the sha256 above belongs to this revision only.
        assert spec.revision in spec.url


def test_pinned_paths_and_addresses():
    flow = config.gguf_spec("4b", "q8_0", "flow")
    assert flow.path == Path("models/rizzo-flow/spark-x2.5-4b-rizzo-flow-lora-q8_0.gguf")
    assert flow.url == (
        f"https://huggingface.co/rizzoaiacademy/rizzo-flow/resolve/{flow.revision}"
        "/spark-x2.5-4b-rizzo-flow-lora-q8_0.gguf"
    )
    base = config.gguf_spec("1.7b", "bf16", "base")
    assert base.path == Path("models/Spark-X2.5-1.7B-GGUF/Spark-X2.5-1.7B.gguf")
    assert base.url == (
        f"https://huggingface.co/XHToken/Spark-X2.5-1.7B-GGUF/resolve/{base.revision}"
        "/Spark-X2.5-1.7B.gguf"
    )


def test_fine_tuned_checkpoints_live_in_the_repositories_of_the_fine_tuned_gguf_files():
    for size, checkpoint in config.FLOW_CHECKPOINTS.items():
        gguf = config.GGUF[(size, "q8_0", "flow")]
        assert (checkpoint.repo, checkpoint.revision) == (gguf.repo, gguf.revision)
        assert checkpoint.path == gguf.path.parent
    assert set(config.FLOW_CHECKPOINTS) == set(config.MODELS)
    assert len(set(config.CHECKPOINT_FILES)) == len(config.CHECKPOINT_FILES)


def test_gguf_spec_is_a_plain_lookup_with_defaults_and_never_falls_back():
    for size in ("4b", "1.7b", "9b"):
        for quant in (None, *config.QUANTS, "q2_k"):
            for variant in (None, *config.VARIANTS, "merged"):
                key = (size, quant or "q8_0", variant or "flow")
                if key in config.GGUF:
                    assert config.gguf_spec(size, quant, variant) is config.GGUF[key]
                else:
                    with pytest.raises(ValueError, match=f"^No {key[2]} GGUF for {size} {key[1]};"):
                        config.gguf_spec(size, quant, variant)


def test_missing_combinations_list_what_exists():
    flow = (
        r"^No flow GGUF for 4b q2_k; flow has: bf16, q4_k_m, q8_0"
        r" \(use --weights base for the original q4_k_m\)$"
    )
    with pytest.raises(ValueError, match=flow):
        config.gguf_spec("4b", "q2_k")
    # Only the fine-tune has a hint to the other weights.
    with pytest.raises(
        ValueError, match=r"^No base GGUF for 1\.7b q2_k; base has: bf16, q4_k_m, q8_0$"
    ):
        config.gguf_spec("1.7b", "q2_k", "base")
    with pytest.raises(ValueError, match=r"^No merged GGUF for 4b q8_0; merged has: $"):
        config.gguf_spec("4b", "q8_0", "merged")
    # Only the files of the size asked for are listed: an unknown size has none.
    with pytest.raises(ValueError, match=r"^No base GGUF for 9b q8_0; base has: $"):
        config.gguf_spec("9b", "q8_0", "base")


def test_checkpoint_path_follows_the_variant():
    assert config.checkpoint_path() == Path("models/rizzo-flow")
    assert config.checkpoint_path("4b", "flow") == Path("models/rizzo-flow")
    assert config.checkpoint_path("1.7b") == Path("models/rizzo-flow-1.7b")
    assert config.checkpoint_path("4b", "base") == Path("models/Spark-X2.5-4B")
    assert config.checkpoint_path("1.7b", "base") == Path("models/Spark-X2.5-1.7B")


# --- Hugging Face token ----------------------------------------------------------------------


def test_no_token_anywhere_is_none(hub):
    assert config.hf_token() is None


def test_environment_wins_over_dotenv_and_login_and_is_stripped(hub, monkeypatch):
    write_dotenv(hub, "HF_TOKEN=from_dotenv\n")
    save_login(hub, "from_login\n")
    assert config.hf_token() == "from_dotenv"  # the file beats the saved login
    monkeypatch.setenv("HF_TOKEN", "  hf_env \n")
    assert config.hf_token() == "hf_env"


def test_the_hub_client_variable_is_an_alias_after_hf_token(hub, monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_alias")
    assert config.hf_token() == "hf_alias"
    monkeypatch.setenv("HF_TOKEN", "hf_main")
    assert config.hf_token() == "hf_main"
    monkeypatch.setenv("HF_TOKEN", "")  # an empty variable is not a token
    assert config.hf_token() == "hf_alias"


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
        ("\ufeffHF_TOKEN=bom\n", "bom"),  # Notepad and PowerShell start a file with a BOM
        ("\ufeff# note\nHF_TOKEN=bom\n", "bom"),
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


@settings(
    max_examples=40,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    token=st.text(string.ascii_letters + string.digits + "_-.=/+", min_size=1, max_size=40),
    quote=st.sampled_from(["", "'", '"']),
    inner=st.text(" \t", max_size=3),  # blanks between the quotes and the token
    pad=st.text(" \t", max_size=3),
    comment=st.sampled_from(["", " # note", "\t# note", "  #note"]),
    newline=st.sampled_from(["\n", "\r\n"]),
    neighbours=st.lists(st.sampled_from(["# note", "OTHER=1", "", "HF_TOKEN_X=y"]), max_size=3),
)
def test_dotenv_token_survives_quotes_padding_comments_and_neighbours(
    hub, token, quote, inner, pad, comment, newline, neighbours
):
    value = f"{quote}{inner}{token}{inner}{quote}"
    lines = [*neighbours, f"{pad}HF_TOKEN{pad}={pad}{value}{pad}{comment}"]
    write_dotenv(hub, newline.join(lines) + newline)
    assert config.hf_token() == token


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


def test_login_token_is_read_last_and_stripped(hub):
    save_login(hub, "  hf_login \n\n")
    assert config.hf_token() == "hf_login"


def test_blank_login_file_is_no_token(hub):
    save_login(hub, " \n")
    assert config.hf_token() is None


@pytest.mark.parametrize("hf_home", [None, ""])
def test_login_token_defaults_to_the_cache_directory_of_the_hub_client(hub, monkeypatch, hf_home):
    home = hub / "home"
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text("hf_home\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    if hf_home is None:
        monkeypatch.delenv("HF_HOME")
    else:
        monkeypatch.setenv("HF_HOME", hf_home)
    assert config.hf_token() == "hf_home"
    # An explicit HF_HOME replaces the default location instead of adding to it.
    monkeypatch.setenv("HF_HOME", str(hub / "elsewhere"))
    assert config.hf_token() is None


# --- downloads ---------------------------------------------------------------------------------


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


def test_download_gguf_fetches_the_pinned_default_with_the_token(hub, monkeypatch, fetched):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    spec = config.gguf_spec()
    assert config.download_gguf() == spec.path
    assert fetched == [
        {
            "url": spec.url,
            "target": spec.path,
            "sha256": spec.sha256,
            "progress": None,
            "token": "hf_secret",
        }
    ]


def test_download_gguf_passes_every_option_through(hub, fetched):
    def progress(name, done, total):
        raise AssertionError("only handed over, never called here")

    destination = hub / "chosen" / "model.gguf"
    spec = config.GGUF[("1.7b", "q4_k_m", "base")]
    # Positional, in the order the command line uses.
    assert config.download_gguf("1.7b", "q4_k_m", str(destination), progress, "base") == destination
    (call,) = fetched
    assert call["url"] == spec.url
    assert call["sha256"] == spec.sha256
    assert call["target"] == destination
    assert isinstance(call["target"], Path)  # a string destination becomes a path
    assert call["progress"] is progress
    assert call["token"] is None


def test_download_gguf_refuses_missing_combinations_before_fetching(hub, fetched):
    with pytest.raises(ValueError, match=r"^No base GGUF for 4b q2_k;"):
        config.download_gguf("4b", "q2_k", variant="base")
    assert fetched == []


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
    seen = script(monkeypatch, Response(payload))
    (hub / "models").mkdir()
    assert config.download_gguf(destination="models") == Path("models") / spec.file
    assert (hub / "models" / spec.file).read_bytes() == payload
    assert sorted(path.name for path in hub.iterdir()) == ["models"]  # no `models.part` beside it
    assert [path.name for path in (hub / "models").iterdir()] == [spec.file]
    assert len(seen) == 1


@pytest.mark.parametrize(("batch_size", "prefill_chunk"), [(1, 1), (4, 512), (16, 2048), (1, 2048)])
def test_the_limits_of_both_backends_include_their_bounds(batch_size, prefill_chunk):
    config.check_limits(batch_size, prefill_chunk)  # nothing to report


@pytest.mark.parametrize(
    ("batch_size", "prefill_chunk"),
    [(0, 512), (-1, 512), (17, 512), (4, 0), (4, -1), (4, 2049), (0, 0), (17, 2049)],
)
def test_the_limits_of_both_backends_refuse_what_is_outside_them(batch_size, prefill_chunk):
    with pytest.raises(ValueError, match=r"^batch_size must be 1–16 and prefill_chunk 1–2048$"):
        config.check_limits(batch_size, prefill_chunk)


@pytest.fixture
def snapshots(monkeypatch):
    """A stand-in for huggingface_hub, which the MLX extra installs and the tests must not need."""
    calls: list[tuple] = []

    def snapshot_download(repo_id, **options):
        calls.append((repo_id, options))
        return "downloaded-directory"

    package = types.ModuleType("huggingface_hub")
    package.__dict__["snapshot_download"] = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", package)
    return calls


def test_download_model_of_the_fine_tune_takes_the_checkpoint_files_only(
    hub, monkeypatch, snapshots
):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    spec = config.FLOW_CHECKPOINTS["4b"]
    assert config.download_model() == "downloaded-directory"
    expected = (
        "rizzoaiacademy/rizzo-flow",
        {
            "revision": spec.revision,
            "local_dir": spec.path,
            "allow_patterns": [*config.CHECKPOINT_FILES, *spec.weights],
            "token": "hf_secret",
        },
    )
    assert snapshots == [expected]
    # The list of files is rebuilt on every call, never grown in place.
    config.download_model()
    assert snapshots[1] == expected
    assert len(config.CHECKPOINT_FILES) == 12


def test_download_model_of_the_small_fine_tune_and_a_chosen_directory(hub, snapshots):
    spec = config.FLOW_CHECKPOINTS["1.7b"]
    config.download_model("elsewhere", "1.7b", "flow")
    ((repo, options),) = snapshots
    assert repo == "rizzoaiacademy/rizzo-flow-1.7b"
    assert options["local_dir"] == "elsewhere"
    assert options["revision"] == spec.revision
    assert options["allow_patterns"] == [*config.CHECKPOINT_FILES, *spec.weights]
    assert options["token"] is None


def test_download_model_of_the_original_weights_is_the_pinned_public_repository(hub, snapshots):
    spec = config.MODELS["1.7b"]
    assert config.download_model(size="1.7b", variant="base") == "downloaded-directory"
    assert snapshots == [
        (
            "XHToken/Spark-X2.5-1.7B",
            {
                "revision": spec.revision,
                "local_dir": spec.path,
                "allow_patterns": [
                    "*.json",
                    "*.jinja",
                    "*.safetensors",
                    "tokenizer.model",
                    "LICENSE",
                    "README.md",
                ],
            },
        )
    ]
    config.download_model("mine", "4b", "base")
    repo, options = snapshots[1]
    assert repo == "XHToken/Spark-X2.5-4B"
    assert options["local_dir"] == "mine"
    assert options["revision"] == config.MODELS["4b"].revision
