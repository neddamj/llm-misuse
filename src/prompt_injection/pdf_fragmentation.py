#!/usr/bin/env python3
"""Create a fragmented-text PDF without third-party dependencies.

The generated PDF renders readable questions for a human, but the underlying
text stream interleaves random hidden glyphs with the visible characters. This
rewrite avoids PyMuPDF by writing a minimal PDF directly with the standard
library and verifying the saved text stream with a small parser.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import string
import subprocess
from dataclasses import dataclass
from pathlib import Path


LETTER_WIDTH = 612
LETTER_HEIGHT = 792
DEFAULT_OUTPUT_FILENAME = "questions.pdf"
DEFAULT_FONT_SIZE = 12
DEFAULT_HEADER_FONT_SIZE = 16
DEFAULT_START_X = 50
DEFAULT_START_Y = 680
DEFAULT_LINE_SPACING = 30
DEFAULT_SEED = 1337
COURIER_GLYPH_WIDTH = 0.6  # Courier is monospaced: 600 units per 1000 em.

DEFAULT_QUESTIONS = (
    "1. Explain the primary causes of World War I.",
    "2. Define the concept of photosynthesis in plants.",
    "3. Solve for x: 2x + 5 = 15.",
)

COMMAND_RE = re.compile(
    r"BT /(?P<font>F[12]) (?P<size>\d+(?:\.\d+)?) Tf "
    r"(?P<r>\d+(?:\.\d+)?) (?P<g>\d+(?:\.\d+)?) (?P<b>\d+(?:\.\d+)?) rg "
    r"1 0 0 1 (?P<x>\d+(?:\.\d+)?) (?P<y>\d+(?:\.\d+)?) Tm "
    r"\((?P<text>(?:\\.|[^\\()])*)\) Tj ET"
)


@dataclass(frozen=True)
class FragmentationConfig:
    output_path: Path
    seed: int = DEFAULT_SEED
    font_size: int = DEFAULT_FONT_SIZE
    header_font_size: int = DEFAULT_HEADER_FONT_SIZE
    start_x: int = DEFAULT_START_X
    start_y: int = DEFAULT_START_Y
    line_spacing: int = DEFAULT_LINE_SPACING
    questions: tuple[str, ...] = DEFAULT_QUESTIONS


def escape_pdf_string(text: str) -> str:
    """Escape a string for a PDF literal string."""
    escaped: list[str] = []
    for char in text:
        if char == "\\":
            escaped.append("\\\\")
        elif char == "(":
            escaped.append("\\(")
        elif char == ")":
            escaped.append("\\)")
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif not (32 <= ord(char) <= 126):
            escaped.append(f"\\{ord(char):03o}")
        else:
            escaped.append(char)
    return "".join(escaped)


def unescape_pdf_string(text: str) -> str:
    """Decode a limited PDF literal string escape set."""
    decoded: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "\\":
            decoded.append(char)
            index += 1
            continue

        index += 1
        if index >= len(text):
            decoded.append("\\")
            break

        escape = text[index]
        if escape in "\\()":
            decoded.append(escape)
            index += 1
        elif escape == "n":
            decoded.append("\n")
            index += 1
        elif escape == "r":
            decoded.append("\r")
            index += 1
        elif escape == "t":
            decoded.append("\t")
            index += 1
        elif escape == "b":
            decoded.append("\b")
            index += 1
        elif escape == "f":
            decoded.append("\f")
            index += 1
        elif escape.isdigit():
            octal_digits = [escape]
            index += 1
            while index < len(text) and len(octal_digits) < 3 and text[index].isdigit():
                octal_digits.append(text[index])
                index += 1
            decoded.append(chr(int("".join(octal_digits), 8)))
        else:
            decoded.append(escape)
            index += 1
    return "".join(decoded)


def glyph_advance(font_size: int) -> float:
    return round(font_size * COURIER_GLYPH_WIDTH, 2)


def random_garbage_char(rng: random.Random) -> str:
    return rng.choice(string.ascii_letters + string.digits)


def text_command(
    font_alias: str,
    font_size: int,
    x: float,
    y: float,
    text: str,
    color: tuple[int, int, int],
) -> str:
    escaped_text = escape_pdf_string(text)
    r, g, b = color
    return (
        f"BT /{font_alias} {font_size} Tf "
        f"{r} {g} {b} rg "
        f"1 0 0 1 {x:.2f} {y:.2f} Tm "
        f"({escaped_text}) Tj ET"
    )


def build_fragmented_line(
    text: str,
    x: float,
    y: float,
    font_size: int,
    rng: random.Random,
) -> tuple[list[str], str]:
    """Create PDF text commands and a preview of the logical text stream."""
    commands: list[str] = []
    logical_stream: list[str] = []
    current_x = x
    advance = glyph_advance(font_size)

    for char in text:
        if char != " ":
            garbage = random_garbage_char(rng)
            commands.append(text_command("F1", font_size, current_x, y, garbage, (1, 1, 1)))
            logical_stream.append(garbage)

        commands.append(text_command("F1", font_size, current_x, y, char, (0, 0, 0)))
        logical_stream.append(char)
        current_x += advance

    return commands, "".join(logical_stream)


def build_content_stream(config: FragmentationConfig) -> tuple[str, str]:
    rng = random.Random(config.seed)
    commands: list[str] = []
    logical_lines: list[str] = []

    header_y = config.start_y + config.line_spacing + 24
    commands.append(
        text_command(
            font_alias="F2",
            font_size=config.header_font_size,
            x=config.start_x,
            y=header_y,
            text="Final Exam - Do Not Cheat",
            color=(0, 0, 0),
        )
    )
    logical_lines.append("Final Exam - Do Not Cheat")

    current_y = config.start_y
    for question in config.questions:
        line_commands, logical_stream = build_fragmented_line(
            text=question,
            x=config.start_x,
            y=current_y,
            font_size=config.font_size,
            rng=rng,
        )
        commands.extend(line_commands)
        logical_lines.append(logical_stream)
        current_y -= config.line_spacing

    return "\n".join(commands) + "\n", "\n".join(logical_lines)


def pdf_object(number: int, body: bytes) -> bytes:
    return f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"


def build_pdf(content_stream: str) -> bytes:
    content_bytes = content_stream.encode("ascii")

    objects = [
        pdf_object(1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        pdf_object(2, b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>"),
        pdf_object(
            3,
            (
                b"<< /Type /Page /Parent 2 0 R "
                b"/MediaBox [0 0 612 792] "
                b"/Resources << /ProcSet [/PDF /Text] /Font << /F1 4 0 R /F2 5 0 R >> >> "
                b"/Contents 6 0 R >>"
            ),
        ),
        pdf_object(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"),
        pdf_object(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"),
        pdf_object(
            6,
            (
                f"<< /Length {len(content_bytes)} >>\n".encode("ascii")
                + b"stream\n"
                + content_bytes
                + b"endstream"
            ),
        ),
    ]

    pdf_bytes = bytearray()
    pdf_bytes.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(obj)

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_bytes.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf_bytes.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf_bytes)


def write_pdf(output_path: Path, content_stream: str) -> None:
    output_path.write_bytes(build_pdf(content_stream))


def extract_logical_stream_from_pdf(pdf_path: Path) -> str:
    """Parse the saved PDF and recover text in the written command order."""
    pdf_text = pdf_path.read_text(encoding="latin-1")
    lines: list[str] = []
    current_y: str | None = None
    current_line: list[str] = []

    for match in COMMAND_RE.finditer(pdf_text):
        y_value = match.group("y")
        if current_y is None:
            current_y = y_value
        elif y_value != current_y:
            lines.append("".join(current_line))
            current_line = []
            current_y = y_value

        current_line.append(unescape_pdf_string(match.group("text")))

    if current_line:
        lines.append("".join(current_line))

    return "\n".join(lines)


def create_fragmented_pdf(config: FragmentationConfig) -> tuple[str, str]:
    content_stream, expected_stream = build_content_stream(config)
    write_pdf(config.output_path, content_stream)
    extracted_stream = extract_logical_stream_from_pdf(config.output_path)
    return expected_stream, extracted_stream


def extract_with_pdftotext(pdf_path: Path) -> str | None:
    """Use pdftotext when available for a real-world extraction check."""
    if shutil.which("pdftotext") is None:
        return None

    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"output PDF path (default: {DEFAULT_OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed for deterministic garbage text (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = FragmentationConfig(output_path=Path(args.output), seed=args.seed)

    print("[*] Generating fragmented PDF...")
    expected_stream, extracted_stream = create_fragmented_pdf(config)
    print(f"[Success] Saved adversarial PDF to: {config.output_path}")

    print("\n--- VERIFICATION STEP ---")
    print("Reading the saved PDF text commands in stream order...")
    print("-" * 30)
    print("RAW DATA STREAM SEEN BY A SIMPLE EXTRACTOR:")
    print("-" * 30)
    print(extracted_stream)
    print("-" * 30)

    if extracted_stream != expected_stream:
        print("[Warning] Extracted stream differed from the generated command preview.")
        return 1

    pdftotext_output = extract_with_pdftotext(config.output_path)
    if pdftotext_output:
        print("\n--- PDFTOTEXT CHECK ---")
        print("Real extractor output from pdftotext:")
        print("-" * 30)
        print(pdftotext_output)
        print("-" * 30)

    print("\n[Final Test Instruction]")
    print("1. Open the generated PDF. It should look like a normal exam.")
    print("2. Select and copy text from it in a PDF viewer.")
    print("3. Compare the pasted text with the verification output above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
