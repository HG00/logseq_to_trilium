# Logseq to Trilium Importer

A Python script to migrate Logseq graphs into Trilium Notes, preserving hierarchy, wikilinks, and metadata.

## Features

* **Page import** with Logseq namespace hierarchy (`project___sub___page` → nested notes)
* **Journal import** into Trilium's native Day Notes calendar system
* **Wikilink resolution** — `[[wikilinks]]` become functional Trilium internal links
* **Stub page creation** for dangling links (tagged `#stub` for easy review)
* **Property extraction** — Logseq `tags::` and `url::` become Trilium labels
* **Code block preservation** — avoids mangling shell `[[ ]]` conditionals

## Prerequisites

* Python 3.10+
* Trilium Notes (tested with TriliumNext)
* An ETAPI token from Trilium (Options → ETAPI → Create new token)

## Installation

```bash
pip install trilium-py mistune
```
## Usage

* Get an ETAPI Token
In Trilium: Menu then Options then ETAPI then Create new token
* Set environment
```bash
export TRILIUM_URL="http://localhost:8080"
export TRILIUM_TOKEN="your_token_here"
```
* If you're on a new trilium installation, click "go to todays journal" to
initialise the calendar/journal - we'll import all journal entries from logseq
here.
* Run the import
```bash
# Dry run
python logseq_to_trilium.py ~/path/to/logseq/graph --dry-run
# Wet run
python logseq_to_trilium.py ~/path/to/logseq/graph
```

## Output
Output tries to match trilium, so journals will be saved in the same way
Logseq allows you to make stub pages just by making a link, we create blank
pages for these in trilium, otherwise they show as missing links
```txt
root
├── Pages
│   ├── SomeNote
│   ├── project
│   │   └── subproject
│   │       └── LeafNote
│   └── StubPage (#stub, #logseq-import)
└── Journal (#calendarRoot)
    └── 2024
        └── 03 - March
            └── 21 - Thursday (#dateNote=2024-03-21)
```
Additional files:
|File |	Purpose |
|-----|-----|
|title_mapping.json | Page title to Trilium noteId mapping (used for wikilink resolution) |
|unresolved_links.json | Any wikilinks that could not be resolved (should be empty if stubs are created) |

## Finding Imported Content

In Trilium's search:

* #stub finds pages created for unresolved wikilinks
* #logseq-import finds stub pages from this import
* #tag=foo finds pages with specific tags (eg `tag:: foo` in logseq)

## Known Limitations
* Markdown conversion is imperfect: Logseq's outliner format (bullet points with tab indentation) doesn't map cleanly to standard Markdown. Most content imports correctly, but complex formatting may need manual cleanup.
* Block references not supported: Logseq ((block-id)) references are not converted.
* Embeds not supported: {{embed [[Page]]}} syntax is not processed.
* Assets not imported: Images and attachments in assets/ are not handled.
* Queries not converted: Logseq datalog queries remain as text.

## License
MIT - do what you like, try and be nice.
