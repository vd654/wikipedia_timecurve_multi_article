"""Shared conversion from MediaWiki source to plain running text."""

import html
import re

try:
    import mwparserfromhell
except ImportError as exc:  # Fail clearly instead of silently using bad regex cleanup.
    raise ImportError(
        "mwparserfromhell fehlt. Bitte `pip install -r requirements.txt` ausführen."
    ) from exc


DROP_TAGS = {
    "ref", "references", "gallery", "math", "score", "timeline", "syntaxhighlight",
    "source", "code", "pre", "imagemap", "mapframe", "maplink", "chem", "graph",
}
MEDIA_NAMESPACES = {"file", "image", "category"}


def _remove_tables(text: str) -> str:
    """Remove MediaWiki tables, including multiline table contents."""
    return re.sub(r"(?ms)^\s*\{\|.*?^\s*\|\}\s*$", " ", text)


def clean_wikitext(text: str) -> str:
    """Return visible prose without templates, citations, tables, or wiki markup."""
    if not text:
        return ""

    text = _remove_tables(text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"__[A-Z][A-Z_]+__", " ", text)
    code = mwparserfromhell.parse(text)

    # Templates include infoboxes, citation templates, navboxes and their nested
    # parameters. Removing the complete node prevents parameter leakage.
    for node in list(code.filter_templates(recursive=True)):
        try:
            code.remove(node, recursive=True)
        except ValueError:
            pass

    # References and non-prose extension tags must be removed with their content.
    for node in list(code.filter_tags(recursive=True)):
        tag = str(node.tag).strip().casefold()
        if tag in DROP_TAGS:
            try:
                code.remove(node, recursive=True)
            except ValueError:
                pass

    # Drop files/categories completely. For ordinary links, strip_code below keeps
    # the displayed label (or target), which is part of the readable sentence.
    for node in list(code.filter_wikilinks(recursive=True)):
        target = str(node.title).strip()
        namespace = target.split(":", 1)[0].casefold() if ":" in target else ""
        if namespace in MEDIA_NAMESPACES:
            try:
                code.remove(node, recursive=True)
            except ValueError:
                pass

    plain = code.strip_code(normalize=True, collapse=True, keep_template_params=False)
    plain = html.unescape(plain)

    # Remove list/table residue and standalone metadata lines, while keeping their
    # readable text when it is sentence-like.
    lines = []
    for line in plain.splitlines():
        line = re.sub(r"^\s*[|!]+\s*", "", line)
        line = re.sub(r"^\s*[*#:;]+\s*", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or re.match(r"^(category|file|image)\s*:", line, re.I):
            continue
        lines.append(line)

    plain = " ".join(lines)
    plain = re.sub(r"https?://\S+|www\.\S+", " ", plain)
    plain = re.sub(r"\s+([,.;:!?])", r"\1", plain)
    plain = re.sub(r"\s+", " ", plain)
    return plain.strip()


def extract_infobox_text(text: str) -> str:
    """Return readable field names and values from article infoboxes."""
    code = mwparserfromhell.parse(text or "")
    fields = []
    for template in code.filter_templates(recursive=False):
        if "infobox" not in str(template.name).casefold():
            continue
        for parameter in template.params:
            name = re.sub(r"[_\s]+", " ", str(parameter.name)).strip()
            value = clean_wikitext(str(parameter.value))
            if name and value:
                fields.append(f"{name}: {value}")
    return " ".join(fields)


def extract_plain_sections(text: str) -> list[dict]:
    """Return cleaned prose grouped by MediaWiki heading.

    Offsets refer to the concatenated cleaned text, not the raw Wikitext.
    """
    heading_re = re.compile(r"(?m)^\s*(={2,6})\s*(.*?)\s*\1\s*$")
    matches = list(heading_re.finditer(text or ""))
    parts = []
    cursor = 0
    heading = "Introduction"
    for match in matches:
        body = clean_wikitext((text or "")[cursor:match.start()])
        if body:
            parts.append((heading, body))
        heading = clean_wikitext(match.group(2)) or "Untitled section"
        cursor = match.end()
    body = clean_wikitext((text or "")[cursor:])
    if body:
        parts.append((heading, body))

    sections = []
    char_cursor = 0
    for index, (section_heading, section_text) in enumerate(parts):
        start = char_cursor
        end = start + len(section_text)
        sections.append({
            "section_index": index,
            "section_heading": section_heading,
            "text": section_text,
            "char_start": start,
            "char_end": end,
        })
        char_cursor = end + 1
    return sections
