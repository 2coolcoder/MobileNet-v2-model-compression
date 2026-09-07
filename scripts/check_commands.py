"""Check that every command printed in the docs actually parses.

README.md, REPORT.md and scripts/run_all.sh all quote exact commands, and the
assignment grades reproducibility on those commands being right.  Flags drift
as a codebase changes, and a stale `--flag` in a README is invisible until
someone tries to run it.  This script extracts every ``python ...`` line from
the docs and feeds its arguments to that script's real argparse parser, so a
renamed or removed option fails here instead of failing for the reader.

    python scripts/check_commands.py        # exits non-zero if any command is stale
"""
import contextlib
import io
import pathlib
import re
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCES = ("scripts/run_all.sh", "README.md", "REPORT.md")


def collect_commands():
    """Every distinct ``python ...`` invocation quoted in the docs."""
    commands, seen = [], set()
    for name in SOURCES:
        path = ROOT / name
        if not path.exists():
            continue
        text = re.sub(r"\\\n\s*", " ", path.read_text())      # join continuations
        for line in text.splitlines():
            line = line.strip().split("#")[0].strip()
            if not line.startswith("python "):
                continue
            for piece in re.split(r"\s*&&\s*", line):          # split shell chains
                piece = piece.strip()
                if piece.startswith("python ") and piece not in seen:
                    seen.add(piece)
                    commands.append((name, piece))
    return commands


def parse_with_script(script: str, argv: list[str]):
    """Run ``script``'s own parser over ``argv``; raises SystemExit on a bad flag."""
    if script == "test.py":
        import test as test_module
        test_module.build_parser().parse_args(argv)
        return
    module_name = script[:-3]
    module = __import__(module_name)
    old = sys.argv
    sys.argv = ["prog"] + argv
    try:
        module.parse_args()
    finally:
        sys.argv = old


INTROSPECTABLE = {"train.py", "test.py", "sweep.py", "analyze.py",
                  "export.py", "upload_sweep.py"}


def main() -> int:
    ok = failed = skipped = 0
    for source, command in collect_commands():
        parts = shlex.split(command)
        if len(parts) < 2 or parts[1] == "-m":
            skipped += 1
            print(f"SKIP {command}")
            continue
        script, argv = parts[1], parts[2:]
        if script.endswith("check_commands.py"):
            skipped += 1
            print(f"SKIP {command}   (this checker itself)")
            continue
        if script not in INTROSPECTABLE:
            skipped += 1
            print(f"SKIP {command}   (no module-level parse_args to introspect)")
            continue
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                parse_with_script(script, argv)
            ok += 1
            print(f"OK   {command}")
        except SystemExit as exc:
            failed += 1
            print(f"FAIL {command}\n     bad or missing flag (exit {exc.code}) "
                  f"-- quoted in {source}")
        except Exception as exc:                                # noqa: BLE001
            failed += 1
            print(f"FAIL {command}\n     {type(exc).__name__}: {exc} "
                  f"-- quoted in {source}")

    print(f"\n{ok} ok, {failed} stale, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
