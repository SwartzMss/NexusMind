from __future__ import annotations

import unicodedata

import pytest

from nexusmind import (
    LexicalAnalyzer,
    UnicodeCJKLexicalAnalyzer,
    WhitespaceLexicalAnalyzer,
)
from nexusmind.lexical_analysis import __all__ as lexical_analysis_exports


def test_analyzers_are_public_protocol_implementations() -> None:
    whitespace: LexicalAnalyzer = WhitespaceLexicalAnalyzer()
    unicode_cjk: LexicalAnalyzer = UnicodeCJKLexicalAnalyzer()

    assert whitespace.analyze("One TWO") == ("one", "two")
    assert unicode_cjk.analyze("One TWO") == ("one", "two")
    assert set(lexical_analysis_exports) == {
        "LexicalAnalyzer",
        "UnicodeCJKLexicalAnalyzer",
        "WhitespaceLexicalAnalyzer",
    }


def test_whitespace_analyzer_preserves_exact_legacy_behavior() -> None:
    analyzer = WhitespaceLexicalAnalyzer()

    assert analyzer.analyze(" Android Binder,  STRAßE\n") == (
        "android",
        "binder,",
        "strasse",
    )
    assert analyzer.analyze("\t\n ") == ()


def test_unicode_analyzer_normalizes_and_extracts_words() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()

    assert analyzer.analyze("Ａｎｄｒｏｉｄ Binder， IPC １２") == (
        "android",
        "binder",
        "ipc",
        "12",
    )
    assert analyzer.analyze("STRAßE cafÉ") == ("strasse", "café")


def test_unicode_analyzer_emits_overlapping_han_bigrams_and_singletons() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()

    assert analyzer.analyze("知识库安全检索") == (
        "知识",
        "识库",
        "库安",
        "安全",
        "全检",
        "检索",
    )
    assert analyzer.analyze("安") == ("安",)


def test_unicode_analyzer_separates_han_non_han_and_other_boundaries() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()

    assert analyzer.analyze("Android 14支持Binder通信") == (
        "android",
        "14",
        "支持",
        "binder",
        "通信",
    )
    assert analyzer.analyze("alpha-beta/delta©omega") == (
        "alpha",
        "beta",
        "delta",
        "omega",
    )


def test_combining_marks_are_boundaries_after_nfkc() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()

    # U+0301 does not compose with x under NFKC, so it separates word runs.
    assert analyzer.analyze("x\u0301y \u0301 z") == ("x", "y", "z")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\u3400\u4dbf", ("\u3400\u4dbf",)),
        ("\u4e00\u9fa5", ("\u4e00\u9fa5",)),
        ("\U00020000\U0002a6d6", ("\U00020000\U0002a6d6",)),
        ("\U00030000\U0003134a", ("\U00030000\U0003134a",)),
        (
            "\uf900\ufa6d",
            (unicodedata.normalize("NFKC", "\uf900\ufa6d"),),
        ),
    ],
)
def test_representative_han_range_boundaries(
    text: str, expected: tuple[str, ...]
) -> None:
    assert UnicodeCJKLexicalAnalyzer().analyze(text) == expected


def test_code_points_outside_han_ranges_are_not_treated_as_han() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()

    assert analyzer.analyze("\u4dc0B \U0002fa20C") == ("b", "c")


def test_unassigned_code_points_inside_han_blocks_are_boundaries() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()

    # U+FA6E is unassigned across the Unicode databases in supported Pythons.
    assert unicodedata.category("\ufa6e") == "Cn"
    assert analyzer.analyze("a\ufa6eb") == ("a", "b")


def test_results_are_stable_tuples_without_empty_tokens() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()
    text = "  A,,,\u5b89\u5168\x00B  "

    first = analyzer.analyze(text)
    second = analyzer.analyze(text)

    assert first == second == ("a", "安全", "b")
    assert type(first) is tuple
    assert all(token for token in first)


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "analyzer", [WhitespaceLexicalAnalyzer(), UnicodeCJKLexicalAnalyzer()]
)
@pytest.mark.parametrize("invalid", [None, 1, b"text", _StringSubclass("text")])
def test_analyzers_require_exact_plain_str(analyzer: LexicalAnalyzer, invalid: object) -> None:
    with pytest.raises(TypeError, match="exact str"):
        analyzer.analyze(invalid)  # type: ignore[arg-type]
