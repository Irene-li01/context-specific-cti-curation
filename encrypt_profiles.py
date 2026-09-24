"""
encrypt_profiles.py — One-time encryption of org profile files.
===============================================================

Run this once on the server after setting CCTI_ENCRYPTION_KEY in .env
to encrypt all files in Profiling/Instances/.

Usage:
    python encrypt_profiles.py

The recommendation engine (recommendation_engine.py) already uses
decrypt_json() when loading profiles, so it works transparently
whether or not the files are encrypted.

To generate a key (if you haven't already):
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from crypto_utils import encrypt_file, encryption_enabled

PROFILES_DIR = ROOT / "Profiling" / "Instances"


def main():
    if not encryption_enabled():
        print("ERROR — CCTI_ENCRYPTION_KEY is not set. Set it in .env before running.")
        sys.exit(1)

    if not PROFILES_DIR.is_dir():
        print(f"ERROR — Profiles directory not found: {PROFILES_DIR}")
        sys.exit(1)

    profiles = sorted(PROFILES_DIR.glob("*.json"))
    if not profiles:
        print("No profile files found.")
        sys.exit(0)

    print(f"Encrypting {len(profiles)} org profile(s) in {PROFILES_DIR}...\n")
    for path in profiles:
        encrypt_file(path)
        print(f"  [OK] {path.name}")

    print(f"\nDone. {len(profiles)} profile(s) encrypted.")
    print("The recommendation engine will decrypt them automatically at runtime.")


if __name__ == "__main__":
    main()
