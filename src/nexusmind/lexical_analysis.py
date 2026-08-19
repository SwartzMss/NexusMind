"""Dependency-free lexical analysis contracts and implementations.

Results are pinned across supported Python versions: normalization and
ordinary character categories use Python's frozen UCD 3.2 database, and Han
recognition uses the explicitly assigned Unicode 14.0 repertoire below.
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
    """Preserve legacy whitespace splitting with Unicode case folding."""

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

_UNICODE = unicodedata.ucd_3_2_0


def _is_han(character: str) -> bool:
    code_point = ord(character)
    return any(start <= code_point <= end for start, end in _HAN_RANGES)


@dataclass(frozen=True, slots=True)
class UnicodeCJKLexicalAnalyzer:
    """Extract Unicode letter/number runs and overlapping Han bigrams.

    NFKC normalization is applied first using the frozen UCD 3.2 database.
    Its Unicode categories starting with ``L`` or ``N`` form ordinary word
    runs, while punctuation, symbols, whitespace, marks, controls, and
    unassigned characters are boundaries. Han recognition is pinned to the
    assigned Unicode 14.0 intervals, independently of the runtime UCD.
    Han ideographs form separate runs and are emitted as overlapping bigrams,
    except that a one-character run is emitted as a singleton.
    """

    def analyze(self, text: str) -> tuple[str, ...]:
        _require_plain_str(text)
        normalized = _UNICODE.normalize("NFKC", text)
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

        for character in normalized:
            is_han = _is_han(character)
            is_word = is_han or _UNICODE.category(character)[0] in {"L", "N"}
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
        return tuple(tokens)
