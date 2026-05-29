"""Copy tutorials/*.md from the repo root into the Sphinx source tree
so they can be referenced from a toctree.

Run at conf.py import time. Image references like `../docs/foo.png` get
rewritten to `../../foo.png` so the relative-path resolution from
`docs/sphinx/_generated/tutorials/foo.md` lands back in `docs/`.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def copy_tutorials(src_dir: Path, dst_dir: Path) -> int:
    """Copy *.md from src_dir to dst_dir, rewriting image paths.

    Returns the number of files copied.
    """
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Wipe out previously-copied files so deletions in tutorials/ are
    # reflected in the next build.
    for old in dst_dir.glob("*.md"):
        old.unlink()

    count = 0
    for md in sorted(src_dir.glob("*.md")):
        if md.name.upper() == "README.MD":
            # Render the index page differently: turn the markdown table
            # of tutorials into a Sphinx :doc: list embedded in the same
            # README, so the gallery has clickable entries.
            text = _rewrite_readme(md.read_text())
        else:
            text = md.read_text()
        # `../docs/foo.png` -> `../../../foo.png` : depuis le fichier genere
        # docs/sphinx/_generated/tutorials/foo.md il faut remonter 3 niveaux
        # (tutorials -> _generated -> sphinx -> docs) pour retrouver docs/foo.png.
        text = re.sub(r"\(\.\./docs/", "(../../../", text)
        (dst_dir / md.name).write_text(text)
        count += 1
    return count


def _rewrite_readme(text: str) -> str:
    r"""Replace `[Title](NN_file.md)` references with `:doc:\`NN_file\``
    inside the README so the Sphinx version becomes a navigation list.

    Falls back to the plain markdown content if no replacements happen.
    """
    # Markdown table cell : | NN | [Title](NN_file.md) | desc |
    # MyST will render the link as an internal doc reference if we use
    # {doc}`NN_file` instead of [Title](NN_file.md).
    def repl(match: re.Match[str]) -> str:
        label = match.group(1)
        target = match.group(2)
        # Strip the literal .md suffix (NOT rstrip, which removes any
        # of the characters m, d, . that turned "05_euler_2d" into
        # "05_euler_2")
        if target.endswith(".md"):
            target = target[:-3]
        return f"[{label}]({target})"
    return re.sub(r"\[([^\]]+)\]\(([^)]+\.md)\)", repl, text)


def setup_gallery(app_srcdir: str) -> int:
    """Entry point called from conf.py."""
    src = Path(app_srcdir).parent.parent / "tutorials"
    dst = Path(app_srcdir) / "_generated" / "tutorials"
    return copy_tutorials(src, dst)
