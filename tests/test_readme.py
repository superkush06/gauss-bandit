"""Every number the README prints, recomputed from the shipped code.

The README is the page that decides whether anyone trusts the rest of the
repository, and it is the file most likely to drift: an algorithm changes, the
tables in it do not, and the page quietly starts lying. This module removes the
possibility. It parses README.md, pulls out every fenced block that claims to be
program output, runs the thing that produced it, and diffs.

Three kinds of block are covered:

* the ``>>>`` sessions, replayed with :mod:`doctest`;
* the "Sixty seconds" snippet, executed and diffed against the block under it;
* the four ``python3 examples/...`` tables, each re-run as a subprocess and
  required to contain the README's block verbatim as a contiguous run of lines.

Tables are compared exactly — retyping ``8.12`` as ``8.13`` fails the suite.
The ``>>>`` sessions print full float reprs, whose last digit depends on the
platform's ``libm``, so those are compared numerically at ``rel=1e-12``: tight
enough to catch a stale or hand-edited value, loose enough not to fail because
macOS and glibc round ``log`` differently in the seventeenth digit.

Runtime is about 25 seconds, most of it the four example scripts.
"""

from __future__ import annotations

import doctest
import io
import os
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
TEXT = README.read_text()

# Commands the README tells the reader to run, each followed by the block of
# output it claims they will see. Timings are measured on one core.
TABLE_COMMANDS = [
    "python3 examples/optimality.py",                            # ~6 s
    "python3 examples/compare_algos.py --horizon 2000 --runs 20",  # ~3 s
    "python3 examples/exp3_longrun.py",                          # ~4 s
    "python3 examples/execution_router.py",                      # ~10 s
]


@dataclass(frozen=True)
class Block:
    """One fenced block: its info string, its body, and where the fence opens."""

    lang: str
    body: str
    line: int


def fenced_blocks(text: str = TEXT) -> list[Block]:
    """Every ``` -fenced block in the README, in document order."""
    out, lang, start, buf = [], None, 0, []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("```"):
            if lang is None:
                lang, start, buf = line[3:].strip(), lineno, []
            else:
                out.append(Block(lang, "\n".join(buf), start))
                lang = None
        elif lang is not None:
            buf.append(line)
    assert lang is None, "unbalanced code fence in README.md"
    return out


def block_after(line: int, skip: int = 0) -> Block:
    """The fenced block that opens after `line`, skipping `skip` of them first."""
    later = [b for b in fenced_blocks() if b.line > line]
    assert len(later) > skip, f"no fenced block after README.md:{line}"
    return later[skip]


def line_of(needle: str) -> int:
    """Line number of the single README line containing `needle`."""
    hits = [i for i, line in enumerate(TEXT.splitlines(), start=1) if needle in line]
    assert len(hits) == 1, f"expected exactly one README line with {needle!r}, got {hits}"
    return hits[0]


# ---------------------------------------------------------------------------
# >>> sessions
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d+\.\d+(?:e[-+]?\d+)?|-?\d+")


class _FloatTolerantChecker(doctest.OutputChecker):
    """Exact text, except that two numbers may differ in the last few digits.

    ``lai_robbins_lower_bound`` bottoms out in ``math.log``, so its full repr is
    not bit-identical across C libraries. Anything that shifts a value by more
    than 1e-12 relative is a real change and still fails.
    """

    def check_output(self, want: str, got: str, optionflags: int) -> bool:
        if super().check_output(want, got, optionflags):
            return True
        want_nums, got_nums = _NUMBER.findall(want), _NUMBER.findall(got)
        if len(want_nums) != len(got_nums) or not want_nums:
            return False
        if _NUMBER.sub("#", want).strip() != _NUMBER.sub("#", got).strip():
            return False
        return all(
            abs(float(w) - float(g)) <= 1e-12 * max(abs(float(w)), abs(float(g)), 1e-300)
            for w, g in zip(want_nums, got_nums, strict=True)
        )


def doctest_blocks() -> list[Block]:
    return [b for b in fenced_blocks() if ">>>" in b.body]


def test_readme_has_the_doctest_sessions_we_think_it_has():
    """Guard the guard: if the ``>>>`` blocks are renamed away, say so loudly
    rather than passing an empty suite."""
    assert len(doctest_blocks()) == 2


@pytest.mark.parametrize("block", doctest_blocks(), ids=lambda b: f"L{b.line}")
def test_readme_python_sessions_replay(block: Block):
    """Each ``>>>`` line in the README, re-executed against the shipped code."""
    parser = doctest.DocTestParser()
    test = parser.get_doctest(block.body + "\n", {}, f"README.md:{block.line}", str(README),
                              block.line)
    runner = doctest.DocTestRunner(checker=_FloatTolerantChecker(),
                                   optionflags=doctest.NORMALIZE_WHITESPACE)
    log = io.StringIO()
    runner.run(test, out=log.write)
    assert runner.failures == 0, log.getvalue()
    assert runner.tries > 0


# ---------------------------------------------------------------------------
# the "Sixty seconds" snippet
# ---------------------------------------------------------------------------

def test_sixty_seconds_snippet_prints_the_block_under_it():
    """The quickstart is the first thing anyone runs. Its three printed numbers
    are recomputed here from the snippet as written, imports included."""
    heading = line_of("## Sixty seconds")
    snippet, expected = block_after(heading), block_after(heading, skip=1)
    assert snippet.lang == "python" and ">>>" not in snippet.body

    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(snippet.body, "README.md#quickstart", "exec"), {"__name__": "__main__"})

    assert buf.getvalue().rstrip("\n").splitlines() == expected.body.splitlines()


# ---------------------------------------------------------------------------
# the example tables
# ---------------------------------------------------------------------------

def _contiguous_index(haystack: list[str], needle: list[str]) -> int:
    """First index where `needle` appears as a contiguous run of `haystack`."""
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


@pytest.mark.parametrize("command", TABLE_COMMANDS, ids=lambda c: shlex.split(c)[1].split("/")[-1])
def test_example_scripts_still_print_their_readme_tables(command: str):
    """Run the command the README prints and require the README's block to come
    back verbatim.

    The README quotes an excerpt (the table, not the surrounding prose), so the
    check is containment as a contiguous run of lines rather than full equality.
    Every one of these scripts is seeded, so 'verbatim' is a fair demand.
    """
    expected = block_after(line_of(f"`{command}`"))
    argv = shlex.split(command)
    assert argv[0] == "python3"
    # The scripts import `bandit` from the repository root, and a script run
    # from examples/ doesn't get that directory on sys.path unless the package
    # is installed. Putting ROOT on PYTHONPATH pins the test to the shipped
    # tree either way, installed or not.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run([sys.executable, *argv[1:]], cwd=ROOT, capture_output=True,
                          text=True, timeout=600, env=env)
    assert proc.returncode == 0, proc.stderr

    actual = proc.stdout.splitlines()
    want = expected.body.splitlines()
    at = _contiguous_index(actual, want)
    if at < 0:
        diff = "\n".join(f"  README | {w}\n  actual | {a}"
                         for w, a in zip(want, actual, strict=False) if w != a)
        pytest.fail(f"`{command}` no longer prints its README block\n{diff or proc.stdout}")
