"""Utility functions for text processing and normalisation."""

import unicodedata
import pandas as pd


def normalize_name(s) -> str:
    """
    Normalise a name string for comparison by converting to lowercase.

    Args:
        s (str or pd.NA): The name string to normalise.

    Returns:
        str: The normalised name string, or empty string if input is NA.
    """
    return "" if pd.isna(s) else str(s).strip().lower()


def normalize_for_matching(text: str) -> str:
    """
    Normalise text for matching by removing accents and converting to lowercase.

    Args:
        text (str): The text to normalise.

    Returns:
        str: The normalised text with accents removed and lowercase.
    """
    normalised = unicodedata.normalize("NFD", text)
    ascii_text = "".join(c for c in normalised if unicodedata.category(c) != "Mn")
    return ascii_text.lower().strip()