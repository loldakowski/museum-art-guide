#!/usr/bin/env python
"""Check that the numbers in README.md come from the committed result files.

Every percentage, every value with a unit (ms, s, MB, points) and every decimal written 0.xxx in the README must equal,
at the precision written, some value stored in a git-tracked JSON or summary table (.md) under results/ (a fraction may be written as a
percentage: 0.759 -> 75.9 %; a difference may be written without its sign: -0.054 -> 5.4 points).
Exits with status 1 and lists the numbers that match nothing.
    python scripts/check_readme_numbers.py [--readme README.md] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

# Percentages that are a setting, not a result.
NOT_RESULTS = {"95"}  # 95 % confidence intervals

PATTERN = re.compile(r"(?<![\w.])([+−-]?\d+(?:\.\d+)?)(?:\s| )*(%|ms\b|s\b|MB\b|points?\b)"
                     r"|(?<![\w.])(0\.\d{2,4})(?![\d%])")


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def stored_values() -> list[tuple[str, float]]:
    """Result values only: per-step training histories and provenance blocks are left out (thousands of values
    that would match any number by chance)."""
    files = subprocess.run(["git", "ls-files", "results/*.json", "results/**/*.json"], capture_output=True, text=True,
                           check=True).stdout.split()
    files = [f for f in files if "histor" not in f]
    values = [(f + p, v) for f in files for p, v in leaves(json.load(open(f, encoding="utf-8")))
              if ".provenance." not in p]
    # The generated summary.md tables hold the means over queries, which no JSON stores.
    for f in subprocess.run(["git", "ls-files", "results/*.md", "results/**/*.md"], capture_output=True, text=True,
                            check=True).stdout.split():
        for line_no, line in enumerate(open(f, encoding="utf-8"), 1):
            values += [(f"{f}:{line_no}", float(x)) for x in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", line)]
    return values


def matches(text: str, unit: str, values) -> list[str]:
    d = len(text.split(".")[1]) if "." in text else 0
    target = text.lstrip("+-−")
    out = []
    for where, v in values:
        candidates = [abs(v)]
        if unit in ("%", "points"):
            candidates.append(abs(v) * 100)
        if any(f"{c:.{d}f}" == target for c in candidates):
            out.append(where)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--readme", default="README.md")
    p.add_argument("--verbose", action="store_true", help="print the first source of every number")
    a = p.parse_args()
    values = stored_values()
    text = open(a.readme, encoding="utf-8").read()
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # commands, not results
    missing, n = [], 0
    for line_no, line in enumerate(text.splitlines(), 1):
        for m in PATTERN.finditer(line):
            number, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), "")
            unit = "points" if unit.startswith("point") else unit
            if unit == "%" and number in NOT_RESULTS:
                continue
            n += 1
            found = matches(number, unit, values)
            if not found:
                missing.append(f"line {line_no}: {number} {unit}  <- {line.strip()[:90]}")
            elif a.verbose:
                print(f"{number:>9} {unit:6} {found[0]}  (+{len(found) - 1} more)")
    print(f"{n} numbers checked against {len(values)} values in committed results files: {len(missing)} not found")
    for m in missing:
        print("  " + m)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
