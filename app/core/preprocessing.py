import re


def clean_prompt(prompt: str) -> str:
    """Preprocess and normalize an input prompt before routing, token estimation, or caching.

    Performs the following cleaning operations:
    1. Normalizes line endings (converts Windows \\r\\n and legacy \\r to \\n).
    2. Strips unprintable control characters (retaining \\n, \\t, and standard printable text).
    3. Collapses three or more consecutive newline characters (\\n\\n\\n+) to a maximum of two (\\n\\n).
    4. Strips leading and trailing whitespace.
    """
    if not prompt:
        return ""

    # Normalize line endings to standard Unix newline
    normalized = prompt.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unprintable / non-ascii control characters (excluding standard whitespace \t and \n)
    normalized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", normalized)

    # Reduce multiple consecutive newline characters (\n\n\n+) to at most two (\n\n)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # Strip leading and trailing whitespace
    return normalized.strip()
