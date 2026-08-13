import re
from pathlib import Path


def sanitize_filename(title: str) -> str:
    """
    Convert an arbitrary title into a safe filename.
    Works well for Windows.
    """

    # Remove characters that Windows does not allow in filenames
    filename = re.sub(r'[<>:"/\\|?*]', '', title)

    # Replace whitespace with underscores
    filename = re.sub(r'\s+', '_', filename.strip())

    # Remove trailing dots/spaces
    filename = filename.rstrip('. ')

    # Make sure we don't end up with an empty filename
    if not filename:
        filename = "generated_blog"

    return f"{filename.lower()}.md"
