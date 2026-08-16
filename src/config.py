"""
Configuration helpers for the Wikipedia Time Curve pipeline.

The defaults can be overridden from the command line, so the same project can
be used for many Wikipedia articles without editing this file.
"""

import argparse
import os
import re
import unicodedata
from hashlib import sha256
from urllib.parse import quote


# Default values used when no command-line arguments are provided.
DEFAULT_ARTICLE = "Palestine"
DEFAULT_DATE_START = "2003-06-06"
DEFAULT_DATE_END = "2015-12-31"
DEFAULT_SAMPLE_STRATEGY = "all"
DEFAULT_MIN_REVISION_CHARS = 500


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def article_slug(article: str) -> str:
    """
    Convert a Wikipedia article title into a stable folder/file name.

    Examples:
        "Climate change" -> "Climate_change"
        "Israel-Palestine relations" -> "Israel-Palestine_relations"
    """
    # MediaWiki treats spaces and underscores as equivalent. Everything else is
    # encoded reversibly instead of being discarded: deleting non-ASCII
    # characters made unrelated titles such as "日本" and "中国" collide in the
    # same ``data/article`` directory.
    canonical = unicodedata.normalize("NFKC", article).strip().replace(" ", "_")
    canonical = re.sub(r"_+", "_", canonical)
    slug = quote(canonical, safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")

    # Empty and dot-only path components are not safe directory names. The hash
    # keeps even these exceptional inputs deterministic and collision-resistant.
    if not slug or slug in {".", ".."}:
        digest = sha256(canonical.encode("utf-8")).hexdigest()[:12]
        return f"article-{digest}"
    return slug


def article_paths(article: str) -> dict[str, str]:
    slug = article_slug(article)
    article_data_dir = os.path.join(DATA_DIR, slug)
    article_output_dir = os.path.join(OUTPUT_DIR, slug)

    return {
        "slug": slug,
        "data_dir": article_data_dir,
        "output_dir": article_output_dir,
        "xml_path": os.path.join(article_data_dir, f"{slug}.xml"),
        "output_basename": f"{slug}_bert_timecurve",
    }


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--article",
        default=DEFAULT_ARTICLE,
        help=f'Wikipedia article title, for example "Climate change" '
             f'(default: "{DEFAULT_ARTICLE}")',
    )
    parser.add_argument(
        "--min-revision-chars",
        type=int,
        default=DEFAULT_MIN_REVISION_CHARS,
        help=f"Skip revisions shorter than this many characters "
             f"(default: {DEFAULT_MIN_REVISION_CHARS})",
    )
    return parser


def add_download_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_args(parser)
    parser.add_argument(
        "--start",
        default=DEFAULT_DATE_START,
        help=f"Start date for revision download, YYYY-MM-DD "
             f"(default: {DEFAULT_DATE_START})",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_DATE_END,
        help=f"End date for revision download, YYYY-MM-DD "
             f"(default: {DEFAULT_DATE_END})",
    )
    return parser


def add_pipeline_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_args(parser)
    parser.add_argument(
        "--start",
        default=None,
        help="Optional inclusive start date, YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional inclusive end date, YYYY-MM-DD",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("weekly", "monthly", "largest", "all"),
        default=DEFAULT_SAMPLE_STRATEGY,
        help=f"Revision sampling strategy (default: {DEFAULT_SAMPLE_STRATEGY})",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional separate output folder and filename prefix",
    )
    return parser
