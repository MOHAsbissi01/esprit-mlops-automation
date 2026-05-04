"""
utils/loader.py — Shared model-loading utilities
=================================================
Provides a generic, thread-safe pickle loader that is available to all
actor modules.  The loader implements a two-level cache:
  1. An in-memory dict (fast path — survives request lifecycle)
  2. FileNotFoundError with a human-readable hint on missing .pkl files

Usage
-----
    from utils.loader import load_pkl

    model        = load_pkl("actor2_mobilites/outputs/xgboost_charge.pkl")
    feature_list = load_pkl("actor2_mobilites/outputs/xgboost_charge_features.pkl")
"""

import logging
import pickle
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("ml_api.utils")

_cache: dict[str, Any] = {}
_lock = threading.Lock()


def load_pkl(model_path: str | Path, *, force_reload: bool = False) -> Any:
    """
    Load and cache a pickle file.

    Parameters
    ----------
    model_path   : str or Path
        Absolute or relative path to the .pkl file.
    force_reload : bool
        If True, bypass the in-memory cache and reload from disk.

    Returns
    -------
    Any
        The deserialized object (sklearn estimator, list, dict, etc.)

    Raises
    ------
    FileNotFoundError
        If the .pkl file does not exist at the given path.
    """
    key = str(model_path)

    if not force_reload and key in _cache:
        return _cache[key]

    with _lock:
        # Double-check inside the lock (avoid race on first load)
        if force_reload or key not in _cache:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Make sure you have run the corresponding actor main.py "
                    "pipeline to generate the .pkl files."
                )
            logger.info(f'"Loading pkl: {path}"')
            with open(path, "rb") as f:
                _cache[key] = pickle.load(f)

    return _cache[key]


def clear_cache(prefix: str | None = None) -> int:
    """
    Clear the in-memory model cache.

    Parameters
    ----------
    prefix : str, optional
        If provided, only removes entries whose key starts with this prefix
        (e.g. 'actor1_ecologique' to invalidate only actor1 models after
        retraining).

    Returns
    -------
    int
        Number of cache entries removed.
    """
    with _lock:
        if prefix is None:
            count = len(_cache)
            _cache.clear()
        else:
            keys_to_remove = [k for k in _cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del _cache[k]
            count = len(keys_to_remove)

    logger.info(f'"Cache cleared: {count} entries removed (prefix={prefix})"')
    return count


def cache_status() -> dict:
    """Return a summary of the currently cached models."""
    return {
        "total_cached": len(_cache),
        "cached_keys": list(_cache.keys()),
    }
