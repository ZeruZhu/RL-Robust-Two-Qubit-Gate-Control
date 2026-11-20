"""Build the standalone method overview and its README preview."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("DPI must be positive")
    return number


def run_checked(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(f"{Path(command[0]).name} failed:\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=positive_int, default=180,
                        help="PNG resolution in dots per inch (default: 180)")
    args = parser.parse_args()

    executables = {name: shutil.which(name) for name in ("pdflatex", "pdftoppm")}
    missing = [name for name, path in executables.items() if path is None]
    if missing:
        parser.exit(1, "Missing tools on PATH: " + ", ".join(missing) + "\n"
                    "Install a LaTeX distribution and Poppler; see the figure README.\n")

    root = Path(__file__).resolve().parents[1]
    figure = root / "figures" / "method_overview"
    source = figure / "source"
    stem = "method_overview"

    try:
        with tempfile.TemporaryDirectory(prefix="rl-gate-overview-") as directory:
            build = Path(directory)
            for _ in range(2):
                run_checked([
                    executables["pdflatex"],
                    "-no-shell-escape",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    f"-output-directory={build}",
                    f"{stem}.tex",
                ], source)
            run_checked([
                executables["pdftoppm"], "-singlefile", "-r", str(args.dpi),
                "-png", str(build / f"{stem}.pdf"), str(build / stem),
            ], source)
            for extension in ("pdf", "png"):
                output = build / f"{stem}.{extension}"
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError(f"Build did not produce a nonempty {extension} file")
            for extension in ("pdf", "png"):
                target = figure / f"{stem}.{extension}"
                shutil.copyfile(build / target.name, target)
                print(target.relative_to(root).as_posix())
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
