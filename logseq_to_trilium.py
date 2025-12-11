#!/usr/bin/env python3
"""
Logseq to Trilium importer.

Imports Logseq pages and journals into Trilium with:
- Namespace hierarchy (project___sub___page -> project > sub > page)
- Wikilink resolution to Trilium internal links
- Native Day Notes calendar integration

Usage:
    export TRILIUM_URL="http://localhost:8080"
    export TRILIUM_TOKEN="your_token_here"
    python logseq_to_trilium.py /path/to/logseq/graph

Or:
    python logseq_to_trilium.py --help
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import unquote

try:
    import mistune
    from trilium_py.client import ETAPI
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install trilium-py mistune")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


ea = None
title_to_noteid = {}
hierarchy_cache = {}
unresolved_links = []
pages_container_id = None


def connect(url: str, token: str):
    """Establish ETAPI connection."""
    global ea
    ea = ETAPI(url, token)
    root = ea.get_note("root")
    log.info(f"Connected to Trilium. Root: {root['title']}")


def create_note(parent_id: str, title: str, content: str = " ") -> str:
    """Create a note and return its noteId."""
    result = ea.create_note(
        parentNoteId=parent_id,
        title=title,
        type="text",
        content=content if content.strip() else " ",
    )
    if "note" not in result:
        log.error(f"Failed to create note '{title}': {result}")
        raise RuntimeError(f"Note creation failed: {result}")
    return result["note"]["noteId"]


def create_note_with_attrs(
    parent_id: str, title: str, content: str, attrs: list
) -> str:
    """Create a note with attributes."""
    note_id = create_note(parent_id, title, content)
    for attr in attrs:
        ea.create_attribute(
            noteId=note_id,
            type="label",
            name=attr["name"],
            value=attr.get("value", ""),
            isInheritable=False,
        )
    return note_id


def get_note_children(note_id: str) -> list:
    """Get child notes of a given note."""
    note = ea.get_note(note_id)
    children = []
    for child_id in note.get("childNoteIds", []):
        try:
            child = ea.get_note(child_id)
            children.append(child)
        except Exception as e:
            log.warning(f"Could not get child {child_id}: {e}")
    return children


def find_child_by_title(parent_id: str, title: str):
    """Find a child note by title under a parent."""
    children = get_note_children(parent_id)
    for child in children:
        if child.get("title") == title:
            return child
    return None


def parse_logseq_properties(content: str) -> tuple:
    """
    Extract Logseq properties from content.
    Returns (properties_dict, remaining_content).

    Properties are lines like:
        tags:: value1, value2, #value3
        url:: https://example.com
    Since I had mixed tags and #tags, it strips the # off
    """
    properties = {}
    lines = content.split("\n")
    content_start = 0

    for i, line in enumerate(lines):
        # Properties are at the start, before any content
        stripped = line.strip()

        # Skip empty lines at start
        if not stripped:
            content_start = i + 1
            continue

        # Check for property pattern with the :: clue
        match = re.match(r"^([\w-]+)::\s*(.*)", stripped)
        if match:
            key = match.group(1).lower()
            value = match.group(2).strip()
            properties[key] = value
            content_start = i + 1
        else:
            # We're prob on content now
            break

    remaining_content = "\n".join(lines[content_start:])
    return properties, remaining_content


def parse_tags(tags_value: str) -> list:
    """
    Parse tags from Logseq tags property.
    Handles: tags:: tag1, tag2, #tag3, [[linked tag]]
    Returns clean list of tag names.
    """
    if not tags_value:
        return []

    tags = []
    for part in tags_value.split(","):
        tag = part.strip()

        if tag.startswith("#"):
            tag = tag[1:]

        if tag.startswith("[[") and tag.endswith("]]"):
            tag = tag[2:-2]

        tag = tag.strip()

        if tag:
            tags.append(tag)

    return tags


def add_note_labels(note_id: str, labels: dict):
    """
    Add labels to a note.
    labels is a dict like {"tag": ["encryption", "gnome"], "url": ["https://..."]}
    """
    for name, values in labels.items():
        if not isinstance(values, list):
            values = [values]
        for value in values:
            ea.create_attribute(
                noteId=note_id,
                type="label",
                name=name,
                value=str(value) if value else "",
                isInheritable=False,
            )


def filename_to_title(filename: str) -> str:
    """Convert filename to readable title."""
    title = filename.replace(".md", "")
    title = unquote(title)
    return title


def parse_namespace(title: str) -> list:
    """Parse Logseq namespace into path components."""
    parts = title.split("___")
    return [p.strip() for p in parts if p.strip()]




def get_or_create_page_hierarchy(parent_id: str, path_parts: list) -> str:
    """Navigate/create hierarchy and return the parent for the leaf note."""
    current_parent = parent_id
    accumulated_path = []

    for part in path_parts[:-1]:
        accumulated_path.append(part)
        path_key = "/".join(accumulated_path)

        if path_key in title_to_noteid:
            current_parent = title_to_noteid[path_key]
        else:
            note_id = create_note(current_parent, part, " ")
            title_to_noteid[path_key] = note_id
            title_to_noteid[part.lower()] = note_id
            log.debug(f"Created intermediate: {path_key} -> {note_id}")
            current_parent = note_id

    return current_parent


def register_title_variants(title: str, note_id: str):
    """Register multiple lookup keys for a note title."""
    title_to_noteid[title] = note_id
    title_to_noteid[title.lower()] = note_id

    slash_version = title.replace("___", "/")
    title_to_noteid[slash_version] = note_id
    title_to_noteid[slash_version.lower()] = note_id

    parts = parse_namespace(title)
    if parts:
        leaf = parts[-1]
        if leaf.lower() not in title_to_noteid:
            title_to_noteid[leaf] = note_id
            title_to_noteid[leaf.lower()] = note_id




def find_calendar_root() -> str:
    """Find the native Journal note with #calendarRoot."""
    results = ea.search_note("note.labels.calendarRoot != null")
    if results.get("results"):
        return results["results"][0]["noteId"]
    raise RuntimeError(
        "No #calendarRoot Journal found. Create a day note in Trilium first."
    )


def get_or_create_year(calendar_root_id: str, date: datetime) -> str:
    """Get or create year note under calendar root."""
    year_str = str(date.year)
    cache_key = f"year_{year_str}"

    if cache_key in hierarchy_cache:
        return hierarchy_cache[cache_key]

    existing = find_child_by_title(calendar_root_id, year_str)
    if existing:
        hierarchy_cache[cache_key] = existing["noteId"]
        return existing["noteId"]

    note_id = create_note_with_attrs(
        parent_id=calendar_root_id,
        title=year_str,
        content=" ",
        attrs=[
            {"name": "dateNote", "value": year_str},
            {"name": "sorted", "value": ""},
        ],
    )
    log.info(f"Created year note: {year_str}")
    hierarchy_cache[cache_key] = note_id
    return note_id


def get_or_create_month(calendar_root_id: str, date: datetime) -> str:
    """Get or create month note under year."""
    year_id = get_or_create_year(calendar_root_id, date)
    month_title = f"{date.month:02d} - {date.strftime('%B')}"
    date_value = f"{date.year}-{date.month:02d}"
    cache_key = f"month_{date_value}"

    if cache_key in hierarchy_cache:
        return hierarchy_cache[cache_key]

    existing = find_child_by_title(year_id, month_title)
    if existing:
        hierarchy_cache[cache_key] = existing["noteId"]
        return existing["noteId"]

    note_id = create_note_with_attrs(
        parent_id=year_id,
        title=month_title,
        content=" ",
        attrs=[
            {"name": "dateNote", "value": date_value},
            {"name": "sorted", "value": ""},
        ],
    )
    log.info(f"Created month note: {month_title}")
    hierarchy_cache[cache_key] = note_id
    return note_id


def get_or_create_day(calendar_root_id: str, date: datetime, content: str) -> str:
    """Get or create day note under month, with content."""
    month_id = get_or_create_month(calendar_root_id, date)
    day_title = f"{date.day:02d} - {date.strftime('%A')}"
    date_value = date.strftime("%Y-%m-%d")

    existing = find_child_by_title(month_id, day_title)
    if existing:
        existing_content = ea.get_note_content(existing["noteId"])
        if content.strip() and content.strip() not in existing_content:
            merged = (
                existing_content
                + "\n<hr/>\n<p><em>Imported from Logseq:</em></p>\n"
                + content
            )
            ea.update_note_content(existing["noteId"], merged)
            log.debug(f"Merged content into existing: {day_title}")
        return existing["noteId"]

    note_id = create_note_with_attrs(
        parent_id=month_id,
        title=day_title,
        content=content,
        attrs=[{"name": "dateNote", "value": date_value}],
    )
    log.debug(f"Created day note: {day_title}")
    return note_id




def create_stub_note(title: str) -> str:
    """Create a stub note for an unresolved wikilink."""
    global pages_container_id

    if not pages_container_id:
        raise RuntimeError("Pages container not set - cannot create stub")

    note_id = create_note(
        pages_container_id,
        title,
        f"<p><em>Stub note created during Logseq import.</em></p>",
    )

    # Add labels
    ea.create_attribute(
        noteId=note_id, type="label", name="stub", value="", isInheritable=False
    )
    ea.create_attribute(
        noteId=note_id,
        type="label",
        name="logseq-import",
        value="",
        isInheritable=False,
    )

    # Register in mapping so future links to same target resolve
    title_to_noteid[title] = note_id
    title_to_noteid[title.lower()] = note_id

    log.info(f"Created stub: {title}")
    return note_id


def resolve_wikilink(target: str, source_note: str):
    """
    Resolve a wikilink target to a Trilium noteId.
    Creates a stub note if target doesn't exist.
    """
    candidates = [
        target,
        target.lower(),
        target.replace(" ", ""),
        target.replace(" ", "").lower(),
        target.replace("/", "___"),
        target.replace("/", "___").lower(),
    ]

    for candidate in candidates:
        if candidate in title_to_noteid:
            return title_to_noteid[candidate], target

    # this is prbably a blank, make a stub
    try:
        note_id = create_stub_note(target)
        return note_id, target
    except Exception as e:
        log.error(f"Failed to create stub for '{target}': {e}")
        unresolved_links.append({"source": source_note, "target": target})
        return None, None


def convert_wikilinks_to_html(content: str, source_note: str) -> str:
    """
    Convert [[wikilinks]] to Trilium internal link HTML.
    Skips content inside code blocks.
    """
    # Preserve fenced code blocks
    code_blocks = []

    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    # Preserve inline code
    inline_codes = []

    def save_inline_code(match):
        inline_codes.append(match.group(0))
        return f"__INLINE_CODE_{len(inline_codes) - 1}__"

    # Save code blocks and inline code before processing wikilinks
    processed = re.sub(r"```[\s\S]*?```", save_code_block, content)
    processed = re.sub(r"`[^`]+`", save_inline_code, processed)

    # Now convert wikilinks (only in non-code content)
    def replace_wikilink(match):
        target = match.group(1)

        # Skip things that will screw up parsing of links
        if any(c in target for c in ["$", "=", "<", ">", ";", "|", "&", "(", ")"]):
            return match.group(0)  # Return unchanged

        if "|" in target:
            target, display = target.split("|", 1)
        else:
            display = target

        note_id, _ = resolve_wikilink(target.strip(), source_note)

        if note_id:
            return f'<a class="reference-link" href="#root/{note_id}">{escape(display)}</a>'
        else:
            return f'<span class="unresolved-link" style="color: #c0392b;">[[ {escape(display)} ]]</span>'

    pattern = r"\[\[([^\]]+)\]\]"
    processed = re.sub(pattern, replace_wikilink, processed)

    # Restore inline code and code blocks
    for i, code in enumerate(inline_codes):
        processed = processed.replace(f"__INLINE_CODE_{i}__", code)
    for i, block in enumerate(code_blocks):
        processed = processed.replace(f"__CODE_BLOCK_{i}__", block)

    return processed


def convert_logseq_to_markdown(content: str) -> str:
    """Convert Logseq outliner format to standard markdown."""
    lines = content.split("\n")
    result = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        if re.match(r"^[\t ]*[a-z-]+::\s*", line):
            match = re.match(r"^([\t ]*)([\w-]+)::\s*(.*)", line)
            if match:
                indent, key, value = match.groups()
                if key in ("collapsed", "id"):
                    continue
                result.append(f"{indent}**{key}:** {value}")
                continue

        stripped = line.lstrip("\t")
        indent_level = len(line) - len(stripped)

        if stripped.startswith("- "):
            content_part = stripped[2:]

            if content_part.startswith("#"):
                result.append(content_part)
            elif not content_part.startswith("|") and not content_part.startswith(
                "```"
            ):
                md_indent = "    " * indent_level
                result.append(f"{md_indent}- {content_part}")
            else:
                result.append(content_part)
        else:
            result.append(line)

    return "\n".join(result)


def escape_code_blocks(content: str) -> str:
    """Escape HTML within fenced code blocks."""

    def replace_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2)
        escaped_code = escape(code)
        return f"```{lang}\n{escaped_code}\n```"

    pattern = r"```(\w*)\n(.*?)```"
    return re.sub(pattern, replace_code_block, content, flags=re.DOTALL)


def escape_inline_code(content: str) -> str:
    """Escape HTML within inline code."""

    def replace_inline(match):
        code = match.group(1)
        return f"`{escape(code)}`"

    pattern = r"`([^`]+)`"
    return re.sub(pattern, replace_inline, content)


def markdown_to_html(md_content: str) -> str:
    """Convert markdown to HTML."""
    markdown = mistune.create_markdown(escape=False, plugins=["table", "strikethrough"])

    processed = escape_code_blocks(md_content)
    processed = escape_inline_code(processed)

    return markdown(processed)


def process_page_content(raw_content: str) -> str:
    """Convert page content: Logseq -> Markdown -> HTML."""
    md = convert_logseq_to_markdown(raw_content)
    html = markdown_to_html(md)
    return html


def process_journal_content(raw_content: str, source_note: str) -> str:
    """Convert journal content with wikilink resolution."""
    with_links = convert_wikilinks_to_html(raw_content, source_note)
    md = convert_logseq_to_markdown(with_links)
    html = markdown_to_html(md)
    return html




def import_pages(pages_dir: Path, pages_parent_id: str) -> int:
    """Import all pages with namespace hierarchy."""
    if not pages_dir.exists():
        log.error(f"Pages directory not found: {pages_dir}")
        return 0

    md_files = sorted(pages_dir.glob("*.md"))
    log.info(f"Found {len(md_files)} page files")

    imported = 0
    for filepath in md_files:
        title = filename_to_title(filepath.name)
        raw_content = filepath.read_text(encoding="utf-8")

        properties, content_without_props = parse_logseq_properties(raw_content)

        try:
            html_content = process_page_content(content_without_props)
        except Exception as e:
            log.error(f"Failed to convert '{title}': {e}")
            html_content = f"<pre>{escape(raw_content)}</pre>"

        path_parts = parse_namespace(title)

        if len(path_parts) > 1:
            parent_id = get_or_create_page_hierarchy(pages_parent_id, path_parts)
            leaf_title = path_parts[-1]
        else:
            parent_id = pages_parent_id
            leaf_title = title

        try:
            note_id = create_note(parent_id, leaf_title, html_content)
            register_title_variants(title, note_id)

            # Add labels from properties
            labels_to_add = {}

            # Handle tags
            if "tags" in properties:
                tags = parse_tags(properties["tags"])
                if tags:
                    labels_to_add["tag"] = tags

            # Handle other useful properties, add more here if needed
            if "url" in properties:
                labels_to_add["url"] = [properties["url"]]

            if "alias" in properties:
                # these dont work the same in trilium but keeping anyway
                # maybe theres a better way using clone
                aliases = [a.strip() for a in properties["alias"].split(",")]
                labels_to_add["alias"] = aliases

            # Add all labels
            if labels_to_add:
                add_note_labels(note_id, labels_to_add)

            imported += 1

            if imported % 100 == 0:
                log.info(f"Progress: {imported} pages imported...")

        except Exception as e:
            log.error(f"Failed to import '{title}': {e}")

    log.info(f"Imported {imported} pages")
    return imported


def import_journals(journals_dir: Path, calendar_root_id: str) -> int:
    """Import all journal files into native Day Notes."""
    if not journals_dir.exists():
        log.error(f"Journals directory not found: {journals_dir}")
        return 0

    md_files = sorted(journals_dir.glob("*.md"))
    log.info(f"Found {len(md_files)} journal files")

    imported = 0
    skipped = 0

    for filepath in md_files:
        title = filename_to_title(filepath.name)
        title_normalised = title.replace("_", "-")

        try:
            date = datetime.strptime(title_normalised, "%Y-%m-%d")
        except ValueError:
            log.warning(f"Could not parse date from '{title}', skipping")
            skipped += 1
            continue

        raw_content = filepath.read_text(encoding="utf-8")

        try:
            html_content = process_journal_content(raw_content, title)
        except Exception as e:
            log.error(f"Failed to convert journal '{title}': {e}")
            html_content = f"<pre>{escape(raw_content)}</pre>"

        try:
            get_or_create_day(calendar_root_id, date, html_content)
            imported += 1

            if imported % 100 == 0:
                log.info(f"Progress: {imported} journals imported...")

        except Exception as e:
            log.error(f"Failed to import journal '{title}': {e}")
            skipped += 1

    log.info(f"Imported {imported} journals, skipped {skipped}")
    return imported




def main():
    global title_to_noteid

    parser = argparse.ArgumentParser(
        description="Import Logseq graph into Trilium Notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  TRILIUM_URL    Trilium server URL (default: http://localhost:8080)
  TRILIUM_TOKEN  ETAPI token (required)

Example:
  export TRILIUM_URL="http://localhost:8080"
  export TRILIUM_TOKEN="your_token_here"
  python logseq_to_trilium.py ~/notes
        """,
    )
    parser.add_argument(
        "logseq_dir",
        type=Path,
        help="Path to Logseq graph directory (contains pages/ and journals/)",
    )
    parser.add_argument(
        "--pages-only", action="store_true", help="Import only pages, skip journals"
    )
    parser.add_argument(
        "--journals-only",
        action="store_true",
        help="Import only journals (requires existing title mapping)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without making changes",
    )

    args = parser.parse_args()

    trilium_url = os.environ.get("TRILIUM_URL", "http://localhost:8080")
    trilium_token = os.environ.get("TRILIUM_TOKEN")

    if not trilium_token:
        log.error("TRILIUM_TOKEN environment variable not set")
        sys.exit(1)

    logseq_dir = args.logseq_dir.expanduser().resolve()
    pages_dir = logseq_dir / "pages"
    journals_dir = logseq_dir / "journals"

    if not logseq_dir.exists():
        log.error(f"Logseq directory not found: {logseq_dir}")
        sys.exit(1)

    if args.dry_run:
        log.info("=== DRY RUN MODE ===")
        if pages_dir.exists():
            log.info(f"Would import {len(list(pages_dir.glob('*.md')))} pages")
        if journals_dir.exists():
            log.info(f"Would import {len(list(journals_dir.glob('*.md')))} journals")
        return

    connect(trilium_url, trilium_token)

    # pages import
    if not args.journals_only:
        if pages_dir.exists():
            global pages_container_id  # Add this
            pages_container_id = create_note("root", "Pages")  # Change this line
            log.info(f"Created Pages container: {pages_container_id}")
            pages_count = import_pages(pages_dir, pages_container_id)
        else:
            log.warning(f"Pages directory not found: {pages_dir}")
            pages_count = 0

        mapping_file = Path("title_mapping.json")
        with open(mapping_file, "w") as f:
            json.dump(title_to_noteid, f, indent=2)
        log.info(f"Title mapping saved to: {mapping_file}")

    # Journals import
    if not args.pages_only:
        # Load mapping if journals-only mode
        if args.journals_only:
            mapping_file = Path("title_mapping.json")
            if mapping_file.exists():
                with open(mapping_file) as f:
                    title_to_noteid = json.load(f)
                log.info(f"Loaded {len(title_to_noteid)} title mappings")
            else:
                log.warning("No title mapping found, wikilinks may not resolve")

            # Find existing Pages container for stubs
            results = ea.search_note('note.title = "Pages"')
            if results.get("results"):
                pages_container_id = results["results"][0]["noteId"]
                log.info(f"Found existing Pages container: {pages_container_id}")

        if journals_dir.exists():
            try:
                calendar_root_id = find_calendar_root()
                log.info(f"Found calendar root: {calendar_root_id}")
            except RuntimeError as e:
                log.error(str(e))
                log.info("Create a day note in Trilium first to initialise the Journal")
                sys.exit(1)

            journals_count = import_journals(journals_dir, calendar_root_id)
        else:
            log.warning(f"Journals directory not found: {journals_dir}")
            journals_count = 0

    # Report
    log.info("=== Import Complete ===")
    if not args.journals_only:
        log.info(f"Pages imported: {pages_count}")
        log.info(f"Title mappings: {len(title_to_noteid)}")
    if not args.pages_only:
        log.info(f"Journals imported: {journals_count}")
        log.info(f"Unresolved wikilinks: {len(unresolved_links)}")

        if unresolved_links:
            unresolved_file = Path("unresolved_links.json")
            with open(unresolved_file, "w") as f:
                json.dump(unresolved_links, f, indent=2)
            log.info(f"Unresolved links saved to: {unresolved_file}")

            log.info("Sample unresolved:")
            seen = set()
            for item in unresolved_links[:10]:
                if item["target"] not in seen:
                    log.info(f"  '{item['target']}' in {item['source']}")
                    seen.add(item["target"])


if __name__ == "__main__":
    main()
