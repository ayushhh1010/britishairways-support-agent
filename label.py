#!/usr/bin/env python
"""Launcher for the labelling tool, so no PYTHONPATH juggling is needed.

    python label.py replies --annotator <name> --limit 40   # rate replies 1-5
    python label.py intents --annotator <name> --blind      # label intents
    python label.py status                                  # what is labelled so far

Run this from the repo root in any terminal. It puts `src/` on the path and forces
UTF-8 so the tweets render on a Windows console, then hands off to the real CLI.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Windows consoles default to cp1252 and these are real tweets, full of emoji.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from support_agent.labeling.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
