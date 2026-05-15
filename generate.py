#!/usr/bin/env python3
"""
generate.py — single source of truth for chain.md / chain.tex / chain.ipynb
TOC and structural verification.

Reads manifest.json. Provides:
  - load_manifest()
  - update_chain_md_toc()
  - update_chain_ipynb_toc()
  - verify_blocks()

Run as a script to refresh TOCs and verify integrity.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(ROOT, "manifest.json")
CHAIN_MD = os.path.join(ROOT, "chain.md")
CHAIN_TEX = os.path.join(ROOT, "chain.tex")
CHAIN_IPYNB = os.path.join(ROOT, "chain.ipynb")
CHAPTERS_DIR = os.path.join(ROOT, "chapters")


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #
def load_manifest() -> dict[str, Any]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _md_anchor(chapter_id: int, title: str) -> str:
    """GitHub-style anchor for '## Chapter N: title'."""
    raw = f"chapter-{chapter_id}-{title}"
    raw = raw.lower()
    raw = re.sub(r"[^a-z0-9\- ]+", "", raw)
    raw = raw.strip().replace(" ", "-")
    raw = re.sub(r"-+", "-", raw)
    return raw


def _grouped_chapters(chapters: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Preserve manifest order, group consecutive chapters by block."""
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for ch in chapters:
        block = ch["block"]
        if not groups or groups[-1][0] != block:
            groups.append((block, []))
        groups[-1][1].append(ch)
    return groups


# --------------------------------------------------------------------------- #
# chain.md TOC                                                                #
# --------------------------------------------------------------------------- #
def _build_md_toc(chapters: list[dict[str, Any]]) -> str:
    lines = ["<!-- TOC START -->", "## Table of contents", ""]
    for block, chs in _grouped_chapters(chapters):
        lines.append(f"### {block}")
        for ch in chs:
            anchor = _md_anchor(ch["id"], ch["title"])
            lines.append(f"- [Chapter {ch['id']}: {ch['title']}](#{anchor})")
        lines.append("")
    lines.append("<!-- TOC END -->")
    return "\n".join(lines)


def update_chain_md_toc() -> None:
    manifest = load_manifest()
    toc = _build_md_toc(manifest["chapters"])
    with open(CHAIN_MD, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(r"<!-- TOC START -->.*?<!-- TOC END -->", re.DOTALL)
    if not pattern.search(text):
        raise RuntimeError("chain.md is missing <!-- TOC START --> / <!-- TOC END --> markers")
    new_text = pattern.sub(toc, text)
    with open(CHAIN_MD, "w", encoding="utf-8") as f:
        f.write(new_text)


# --------------------------------------------------------------------------- #
# chain.ipynb TOC                                                             #
# --------------------------------------------------------------------------- #
def _build_ipynb_toc_lines(chapters: list[dict[str, Any]]) -> list[str]:
    lines = ["## Table of contents\n", "\n"]
    for block, chs in _grouped_chapters(chapters):
        lines.append(f"### {block}\n")
        for ch in chs:
            lines.append(f"- Chapter {ch['id']}: {ch['title']}\n")
        lines.append("\n")
    return lines


def update_chain_ipynb_toc() -> None:
    manifest = load_manifest()
    with open(CHAIN_IPYNB, "r", encoding="utf-8") as f:
        nb = json.load(f)
    toc_lines = _build_ipynb_toc_lines(manifest["chapters"])
    found = False
    for cell in nb["cells"]:
        if cell.get("cell_type") == "markdown" and cell.get("metadata", {}).get("toc") is True:
            cell["source"] = toc_lines
            found = True
            break
    if not found:
        raise RuntimeError(
            "chain.ipynb is missing a markdown cell with metadata {'toc': true}"
        )
    with open(CHAIN_IPYNB, "w", encoding="utf-8") as f:
        f.write(json.dumps(nb, indent=1))
        f.write("\n")


# --------------------------------------------------------------------------- #
# Block verification                                                          #
# --------------------------------------------------------------------------- #
def verify_blocks() -> bool:
    manifest = load_manifest()
    chapter_ids = [ch["id"] for ch in manifest["chapters"]]

    with open(CHAIN_MD, "r", encoding="utf-8") as f:
        md_text = f.read()
    with open(CHAIN_TEX, "r", encoding="utf-8") as f:
        tex_text = f.read()
    with open(CHAIN_IPYNB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    nb_chapter_ids = {
        cell.get("metadata", {}).get("chapter")
        for cell in nb["cells"]
        if cell.get("cell_type") == "markdown"
    }

    ok = True
    for cid in chapter_ids:
        md_start = f"<!-- CHAPTER {cid} START -->" in md_text
        md_end = f"<!-- CHAPTER {cid} END -->" in md_text
        tex_start = f"% --- CHAPTER {cid} START ---" in tex_text
        tex_end = f"% --- CHAPTER {cid} END ---" in tex_text
        nb_present = cid in nb_chapter_ids

        if not (md_start and md_end and tex_start and tex_end and nb_present):
            print(
                f"[verify_blocks] chapter {cid} missing: "
                f"md_start={md_start} md_end={md_end} "
                f"tex_start={tex_start} tex_end={tex_end} "
                f"ipynb_cell={nb_present}"
            )
            ok = False

    if ok:
        print(f"[verify_blocks] OK: all {len(chapter_ids)} chapters present in all 3 forms")
    return ok


# --------------------------------------------------------------------------- #
# Chapter merge from staging files                                            #
# --------------------------------------------------------------------------- #
# Each chapter agent writes to:
#   chapters/ch{NN}/content.md     — body markdown (no START/END markers)
#   chapters/ch{NN}/content.tex    — body LaTeX (no START/END markers, no \section*)
#   chapters/ch{NN}/cells.json     — JSON list of nbformat-4 cells (markdown + code)
# This merge function splices each chapter's content between the existing
# START/END markers in chain.md and chain.tex, and replaces the stub cell
# in chain.ipynb with the cells from cells.json.

def _chapter_dir(cid: int) -> str:
    return os.path.join(CHAPTERS_DIR, f"ch{cid:02d}")


def _read_if_exists(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def merge_chapters() -> None:
    manifest = load_manifest()
    chapters = manifest["chapters"]

    # ---- chain.md ------------------------------------------------------- #
    with open(CHAIN_MD, "r", encoding="utf-8") as f:
        md_text = f.read()
    for ch in chapters:
        cid = ch["id"]
        body = _read_if_exists(os.path.join(_chapter_dir(cid), "content.md"))
        if body is None:
            continue
        anchor_id = _md_anchor(cid, ch["title"])
        block = (
            f"<!-- CHAPTER {cid} START -->\n"
            f"<a id=\"{anchor_id}\"></a>\n"
            f"## Chapter {cid}: {ch['title']}\n\n"
            f"{body.rstrip()}\n\n"
            f"<!-- CHAPTER {cid} END -->"
        )
        pattern = re.compile(
            rf"<!-- CHAPTER {cid} START -->.*?<!-- CHAPTER {cid} END -->",
            re.DOTALL,
        )
        if not pattern.search(md_text):
            print(f"[merge_chapters] chain.md missing markers for chapter {cid}")
            continue
        md_text = pattern.sub(lambda _m, b=block: b, md_text)
    with open(CHAIN_MD, "w", encoding="utf-8") as f:
        f.write(md_text)

    # ---- chain.tex ------------------------------------------------------ #
    with open(CHAIN_TEX, "r", encoding="utf-8") as f:
        tex_text = f.read()
    for ch in chapters:
        cid = ch["id"]
        body = _read_if_exists(os.path.join(_chapter_dir(cid), "content.tex"))
        if body is None:
            continue
        section_title = ch["title"].replace("&", r"\&").replace("_", r"\_")
        block = (
            f"% --- CHAPTER {cid} START ---\n"
            f"\\section*{{Chapter {cid}: {section_title}}}\n"
            f"\\addcontentsline{{toc}}{{section}}{{Chapter {cid}: {section_title}}}\n\n"
            f"{body.rstrip()}\n\n"
            f"% --- CHAPTER {cid} END ---"
        )
        pattern = re.compile(
            rf"% --- CHAPTER {cid} START ---.*?% --- CHAPTER {cid} END ---",
            re.DOTALL,
        )
        if not pattern.search(tex_text):
            print(f"[merge_chapters] chain.tex missing markers for chapter {cid}")
            continue
        tex_text = pattern.sub(lambda _m, b=block: b, tex_text)
    with open(CHAIN_TEX, "w", encoding="utf-8") as f:
        f.write(tex_text)

    # ---- chain.ipynb ---------------------------------------------------- #
    with open(CHAIN_IPYNB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Build a map: chapter_id -> list of cells from cells.json
    new_cells_by_chapter: dict[int, list] = {}
    for ch in chapters:
        cid = ch["id"]
        cells_path = os.path.join(_chapter_dir(cid), "cells.json")
        if not os.path.exists(cells_path):
            continue
        with open(cells_path, "r", encoding="utf-8") as f:
            cells = json.load(f)
        # Stamp every cell with metadata.chapter = cid
        for cell in cells:
            cell.setdefault("metadata", {})
            cell["metadata"]["chapter"] = cid
            cell.setdefault("source", "")
            if cell.get("cell_type") == "code":
                cell.setdefault("execution_count", None)
                cell.setdefault("outputs", [])
            # nbformat requires `source` to be a list of strings or a string
        new_cells_by_chapter[cid] = cells

    if new_cells_by_chapter:
        rebuilt: list = []
        for cell in nb["cells"]:
            cid = cell.get("metadata", {}).get("chapter")
            if cid in new_cells_by_chapter:
                # Splice in the new cells in place of the stub cell.
                # If we've already spliced this chapter (multi-cell run),
                # skip subsequent stubs.
                if new_cells_by_chapter[cid] is not None:
                    rebuilt.extend(new_cells_by_chapter[cid])
                    new_cells_by_chapter[cid] = None  # mark consumed
                # else: drop additional stubs for this chapter
            else:
                rebuilt.append(cell)
        nb["cells"] = rebuilt

    with open(CHAIN_IPYNB, "w", encoding="utf-8") as f:
        f.write(json.dumps(nb, indent=1))
        f.write("\n")


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    merge_chapters()
    print("[generate] chapter content merged from chapters/ch*/")
    update_chain_md_toc()
    print("[generate] chain.md TOC refreshed")
    update_chain_ipynb_toc()
    print("[generate] chain.ipynb TOC refreshed")
    ok = verify_blocks()
    sys.exit(0 if ok else 1)
