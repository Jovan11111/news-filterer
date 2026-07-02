import html

from src.constants.dataset import BROKEN_LEAD_RE, ENCODED_TAG_RE, REAL_TAG_RE, WS_RE


def clean_text(text: str) -> str:
    """Strip HTML (real, encoded and malformed) and collapse whitespace.

    Args:
        text (str): The raw text to clean.

    Returns:
        str: The cleaned, whitespace-collapsed text.
    """
    if not text:
        return ""
    text = REAL_TAG_RE.sub(" ", text)
    text = ENCODED_TAG_RE.sub(" ", text)
    text = BROKEN_LEAD_RE.sub(" ", text)
    text = html.unescape(text)
    # A second pass removes any tags that only became visible after unescaping.
    text = REAL_TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()
