"""
Simple print helpers.
No longer used by other scripts (they print directly now),
but kept in case you want them.
"""


def banner(title):
    """Print a big header with a title."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def box(label, text):
    """Print a labeled block of text."""
    print(f"--- {label} ---")
    print(text.strip())
    print()
