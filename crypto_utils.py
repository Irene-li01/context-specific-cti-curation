"""
crypto_utils.py — At-rest encryption for sensitive CCTI pipeline outputs.
=========================================================================

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256) from the
cryptography library. A single key is stored in the CCTI_ENCRYPTION_KEY
environment variable.

If CCTI_ENCRYPTION_KEY is not set, all functions are no-ops — the pipeline
runs unencrypted (useful for local development). Set it on the server.

Generating a key (run once, store in .env):
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("CryptoUtils")

try:
    from cryptography.fernet import Fernet, InvalidToken
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not installed — encryption disabled.")


def _get_cipher():
    """Return a Fernet cipher if CCTI_ENCRYPTION_KEY is set, else None."""
    if not _CRYPTO_AVAILABLE:
        return None
    key = os.getenv("CCTI_ENCRYPTION_KEY", "").strip()
    if not key:
        return None
    try:
        return Fernet(key.encode())
    except Exception as e:
        logger.error(f"Invalid CCTI_ENCRYPTION_KEY: {e}")
        return None


def encryption_enabled() -> bool:
    return _get_cipher() is not None


def encrypt_file(path: Path) -> None:
    """Encrypt a file in-place. No-op if CCTI_ENCRYPTION_KEY is not set."""
    cipher = _get_cipher()
    if not cipher:
        return
    try:
        data = path.read_bytes()
        path.write_bytes(cipher.encrypt(data))
        logger.info(f"Encrypted: {path.name}")
    except Exception as e:
        logger.error(f"Failed to encrypt {path}: {e}")


def decrypt_file(path: Path) -> bytes:
    """Decrypt a file and return its bytes. Returns raw bytes if no key set."""
    cipher = _get_cipher()
    if not cipher:
        return path.read_bytes()
    try:
        return cipher.decrypt(path.read_bytes())
    except InvalidToken:
        logger.warning(f"{path.name} — decryption failed (wrong key or unencrypted file). Returning raw bytes.")
        return path.read_bytes()
    except Exception as e:
        logger.error(f"Failed to decrypt {path}: {e}")
        return path.read_bytes()


def decrypt_json(path: Path) -> Any:
    """Read, decrypt (if needed), and parse a JSON file."""
    raw = decrypt_file(path)
    return json.loads(raw.decode("utf-8"))
