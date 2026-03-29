"""TenSEAL context utilities for FHE DIWS."""

from __future__ import annotations

from typing import List, Optional
import os


def _get_tenseal():
    try:
        import tenseal as ts
    except ImportError as exc:
        raise ImportError(
            "tenseal is required for FHE DIWS. Install with: pip install tenseal"
        ) from exc
    return ts


def get_tenseal():
    """Return the TenSEAL module with lazy import."""
    return _get_tenseal()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def create_and_save_context(
    server_path: str,
    client_path: str,
    poly_modulus_degree: int = 16384,
    coeff_mod_bit_sizes: Optional[List[int]] = None,
    global_scale_bits: int = 29,
    generate_galois_keys: bool = False,
):
    """Generate CKKS context and save server/client variants."""
    ts = _get_tenseal()
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 29, 29, 29, 29, 29, 29, 29, 60]

    # ckks context for encrypted scalar ops
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = 2**global_scale_bits
    if generate_galois_keys:
        context.generate_galois_keys()

    _ensure_parent_dir(client_path)
    _ensure_parent_dir(server_path)

    # client context keeps secret key
    with open(client_path, "wb") as f:
        f.write(context.serialize(save_secret_key=True))

    # server context drops secret key
    context.make_context_public()
    with open(server_path, "wb") as f:
        f.write(context.serialize())

    return context


def load_context(path: str):
    """Load a TenSEAL context from disk."""
    ts = _get_tenseal()
    with open(path, "rb") as f:
        data = f.read()
    return ts.context_from(data)


def load_client_context(path: str):
    """Load a TenSEAL context with secret key for client-side use.

    Returns the context which can be used for both encryption and decryption.
    """
    return load_context(path)
