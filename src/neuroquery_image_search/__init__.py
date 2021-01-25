"""Search the NeuroQuery dataset (https://neuroquery.org) for studies and
terms with activation patterns similar to an input image.
"""

__all__ = [
    "NeuroQueryImageSearch",
    "studies_to_html_table",
    "terms_to_html_table",
    "results_to_html",
]

from pathlib import Path

__version__ = (
    Path(__file__).parent.joinpath("data", "VERSION.txt").read_text().strip()
)
from ._searching import (
    NeuroQueryImageSearch,
    studies_to_html_table,
    terms_to_html_table,
    results_to_html,
)
