#!/usr/bin/env python3
"""リポジトリ直下から実行可能。`PYTHONPATH` の設定は不要。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from house_prices.cli import main

if __name__ == "__main__":
    main()
