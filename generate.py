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
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    update_chain_md_toc()
    print("[generate] chain.md TOC refreshed")
    update_chain_ipynb_toc()
    print("[generate] chain.ipynb TOC refreshed")
    ok = verify_blocks()
    sys.exit(0 if ok else 1)
