"""Module entrypoint for `python -m linux_studio_effects`."""

from __future__ import annotations

from .tui import main

if __name__ == "__main__":
    raise SystemExit(main())
