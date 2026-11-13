import pytest
from app.core.preprocessing import clean_prompt


class TestPromptPreprocessing:
    """Unit tests verifying prompt normalization and cleaning rules (Contract 3)."""

    def test_strip_leading_and_trailing_whitespace(self):
        raw = "   \t  Hello World!   \n  "
        cleaned = clean_prompt(raw)
        assert cleaned == "Hello World!"

    def test_reduce_consecutive_newlines_to_max_two(self):
        # 3 newlines -> reduced to 2
        raw = "Paragraph 1\n\n\nParagraph 2"
        assert clean_prompt(raw) == "Paragraph 1\n\nParagraph 2"

        # 5 newlines -> reduced to 2
        raw = "Paragraph 1\n\n\n\n\nParagraph 2"
        assert clean_prompt(raw) == "Paragraph 1\n\nParagraph 2"

        # 2 newlines -> remains 2
        raw = "Paragraph 1\n\nParagraph 2"
        assert clean_prompt(raw) == "Paragraph 1\n\nParagraph 2"

        # 1 newline -> remains 1
        raw = "Line 1\nLine 2"
        assert clean_prompt(raw) == "Line 1\nLine 2"

    def test_normalize_windows_crlf_endings(self):
        raw = "Line 1\r\n\r\n\r\n\r\nLine 2"
        cleaned = clean_prompt(raw)
        assert cleaned == "Line 1\n\nLine 2"

    def test_strip_unprintable_control_characters(self):
        raw = "Valid Text \x00\x05\x08 with \x1b hidden \x7f characters."
        cleaned = clean_prompt(raw)
        assert cleaned == "Valid Text  with  hidden  characters."

    def test_empty_and_whitespace_only_prompts(self):
        assert clean_prompt("") == ""
        assert clean_prompt("   \t  \n\n  ") == ""

    def test_preserves_valid_code_formatting(self):
        raw = "def add(a, b):\n    return a + b\n\n\n\nprint(add(2, 3))"
        cleaned = clean_prompt(raw)
        assert cleaned == "def add(a, b):\n    return a + b\n\nprint(add(2, 3))"

