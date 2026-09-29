"""The bilingual pages keep their Italian and English dictionaries in step.

CLAUDE.md: every new UI string goes in both languages. `t()` falls back to the key itself, so a
missing translation shows up in the page as a raw identifier and nothing else notices.
"""

import ast
import json
import re
from typing import Any

import pytest

from rizzo_flow import api

PAGES = {"playground": api.PLAYGROUND, "snake": api.SNAKE}
STRING = r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'"  # a JavaScript string literal
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def read_page(name: str) -> str:
    return PAGES[name].read_text(encoding="utf-8")


def dictionaries(page: str) -> dict[str, dict[str, Any]]:
    """The `const I18N = {...}` literal: language -> key -> string, or list of strings."""
    literal = re.search(r"^const I18N = (\{.*?^\});$", page, re.MULTILINE | re.DOTALL)
    assert literal, "the page has no `const I18N = {...};` block at the start of a line"

    def as_json(match: re.Match[str]) -> str:
        if match["string"]:
            return json.dumps(ast.literal_eval(match["string"]))
        return json.dumps(match["key"]) if match["key"] else ""  # no key: a trailing comma

    # Strings are matched whole, so a comma or a word inside one is never touched.
    return json.loads(
        re.sub(rf"(?P<string>{STRING})|(?P<key>\w+)(?=\s*:)|,(?=\s*[}}\]])", as_json, literal[1])
    )


def used_keys(page: str) -> set[str]:
    keys = set(re.findall(r'data-i18n(?:-html)?="([^"]+)"', page))
    for pairs in re.findall(r'data-i18n-attr="([^"]+)"', page):  # "attribute:key,attribute:key"
        keys.update(pair.split(":")[1] for pair in pairs.split(","))
    keys.update(re.findall(r'\bt\("(\w+)"', page))  # literal keys only: t(variable) is not followed
    return keys


@pytest.mark.parametrize("name", PAGES)
def test_the_two_languages_define_the_same_keys(name):
    i18n = dictionaries(read_page(name))
    assert set(i18n) == {"it", "en"}
    assert i18n["it"].keys() == i18n["en"].keys()


@pytest.mark.parametrize("name", PAGES)
def test_the_two_translations_of_a_key_have_the_same_shape(name):
    i18n = dictionaries(read_page(name))
    for key, italian in i18n["it"].items():
        english = i18n["en"][key]
        if isinstance(italian, list):  # e.g. dangerLevels, indexed by level
            assert isinstance(english, list), key
            assert len(english) == len(italian), key
        else:  # a placeholder missing from one language would render as an empty string
            assert sorted(PLACEHOLDER.findall(english)) == sorted(PLACEHOLDER.findall(italian)), key


@pytest.mark.parametrize("name", PAGES)
def test_every_key_the_page_uses_is_translated(name):
    page = read_page(name)
    keys = used_keys(page)
    assert len(keys) > 40  # the scan finds the page's references
    assert keys <= dictionaries(page)["en"].keys()
