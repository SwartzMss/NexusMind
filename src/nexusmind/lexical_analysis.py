"""Dependency-free lexical analysis contracts and implementations.

Results are pinned to the Unicode 14.0 repertoire across supported Python
versions. Characters first assigned in Unicode 15.0 or 15.1 are boundaries;
the remaining text uses current NFKC, category, and case-folding behavior,
which is stable for already-assigned characters. Han recognition uses the
explicitly assigned Unicode 14.0 repertoire below.
Combining marks are intentionally not part of word runs. After NFKC
normalization, any mark that remains acts as a token boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Protocol


__all__ = [
    "LexicalAnalyzer",
    "UnicodeCJKLexicalAnalyzer",
    "WhitespaceLexicalAnalyzer",
]


class LexicalAnalyzer(Protocol):
    """Convert text into a stable sequence of normalized lexical tokens."""

    def analyze(self, text: str) -> tuple[str, ...]:
        """Return normalized lexical tokens extracted from ``text``."""


def _require_plain_str(text: str) -> None:
    if type(text) is not str:
        raise TypeError("text must be an exact str")


@dataclass(frozen=True, slots=True)
class WhitespaceLexicalAnalyzer:
    """Provide a deterministic whitespace control for benchmark comparisons."""

    def analyze(self, text: str) -> tuple[str, ...]:
        _require_plain_str(text)
        return tuple(token.casefold() for token in text.split())


# Audited against Unicode 14.0 UnicodeData.txt. These are assigned CJK unified
# and compatibility ideographs, not whole blocks; holes and unassigned tails
# are deliberately excluded. U+3007 is the Han ideographic number zero.
_HAN_RANGES: tuple[tuple[int, int], ...] = (
    (0x3007, 0x3007),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B738),
    (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEA1),
    (0x2CEB0, 0x2EBE0),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
)

# Audited from Unicode 15.1 DerivedAge.txt entries with Age=15.0 or 15.1:
# https://www.unicode.org/Public/15.1.0/ucd/DerivedAge.txt
# Filtering these assignments before normalization keeps the effective
# repertoire identical on CPython 3.11 (UCD 14), 3.12 (UCD 15), and 3.13
# (UCD 15.1). Ranges are intentionally vendored: analysis has no network or
# host-version dependency.
_POST_UNICODE_14_RANGES: tuple[tuple[int, int], ...] = (
    (0x0CF3, 0x0CF3),
    (0x0ECE, 0x0ECE),
    (0x2FFC, 0x2FFF),
    (0x31EF, 0x31EF),
    (0x10EFD, 0x10EFF),
    (0x1123F, 0x11241),
    (0x11B00, 0x11B09),
    (0x11F00, 0x11F10),
    (0x11F12, 0x11F3A),
    (0x11F3E, 0x11F59),
    (0x1342F, 0x1342F),
    (0x13439, 0x1343F),
    (0x13440, 0x13455),
    (0x1B132, 0x1B132),
    (0x1B155, 0x1B155),
    (0x1D2C0, 0x1D2D3),
    (0x1DF25, 0x1DF2A),
    (0x1E030, 0x1E06D),
    (0x1E08F, 0x1E08F),
    (0x1E4D0, 0x1E4F9),
    (0x1F6DC, 0x1F6DC),
    (0x1F774, 0x1F776),
    (0x1F77B, 0x1F77F),
    (0x1F7D9, 0x1F7D9),
    (0x1FA75, 0x1FA77),
    (0x1FA87, 0x1FA88),
    (0x1FAAD, 0x1FAAF),
    (0x1FABB, 0x1FABD),
    (0x1FABF, 0x1FABF),
    (0x1FACE, 0x1FACF),
    (0x1FADA, 0x1FADB),
    (0x1FAE8, 0x1FAE8),
    (0x1FAF7, 0x1FAF8),
    (0x2B739, 0x2B739),
    (0x2EBF0, 0x2EE5D),
    (0x31350, 0x323AF),
)


def _is_han(character: str) -> bool:
    code_point = ord(character)
    return any(start <= code_point <= end for start, end in _HAN_RANGES)


def _is_post_unicode_14(character: str) -> bool:
    code_point = ord(character)
    return any(
        start <= code_point <= end for start, end in _POST_UNICODE_14_RANGES
    )


@dataclass(frozen=True, slots=True)
class UnicodeCJKLexicalAnalyzer:
    """Extract Unicode letter/number runs and overlapping Han bigrams.

    Characters assigned after Unicode 14 are boundaries before normalization.
    Each remaining segment receives NFKC normalization. Unicode categories
    starting with ``L`` or ``N`` form ordinary word runs, while punctuation,
    symbols, whitespace, marks, controls, and unassigned characters are
    boundaries. Han recognition is pinned to assigned Unicode 14.0 intervals,
    independently of the runtime UCD.
    Han ideographs form separate runs and are emitted as overlapping bigrams,
    except that a one-character run is emitted as a singleton.
    """

    def analyze(self, text: str) -> tuple[str, ...]:
        _require_plain_str(text)
        tokens: list[str] = []
        run: list[str] = []
        run_is_han: bool | None = None

        def flush() -> None:
            nonlocal run, run_is_han
            if run_is_han:
                if len(run) == 1:
                    tokens.append(run[0].casefold())
                else:
                    tokens.extend(
                        (run[index] + run[index + 1]).casefold()
                        for index in range(len(run) - 1)
                    )
            elif run:
                tokens.append("".join(run).casefold())
            run = []
            run_is_han = None

        def consume(segment: list[str]) -> None:
            nonlocal run, run_is_han
            normalized = unicodedata.normalize("NFKC", "".join(segment))
            for character in normalized:
                is_han = _is_han(character)
                is_word = is_han or unicodedata.category(character)[0] in {
                    "L",
                    "N",
                }
                if not is_word:
                    flush()
                elif run and is_han != run_is_han:
                    flush()
                    run = [character]
                    run_is_han = is_han
                else:
                    run.append(character)
                    run_is_han = is_han
            flush()

        segment: list[str] = []
        for character in text:
            if _is_post_unicode_14(character):
                consume(segment)
                segment = []
            else:
                segment.append(character)

        consume(segment)
        return tuple(tokens)
