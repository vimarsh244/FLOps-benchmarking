"""OpenFHE context utilities for FHE DIWS."""

from __future__ import annotations

from typing import List, Optional
import os


def _get_openfhe():
    try:
        import openfhe as fhe
    except ImportError as exc:
        raise ImportError(
            "openfhe is required for FHE DIWS. Install with: pip install openfhe"
        ) from exc
    return fhe


def get_openfhe():
    """Return the OpenFHE module with lazy import."""
    return _get_openfhe()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _create_context(
    *,
    poly_modulus_degree: int,
    coeff_mod_bit_sizes: Optional[List[int]],
    global_scale_bits: int,
):
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 29, 29, 29, 29, 29, 29, 29, 60]

    fhe = _get_openfhe()
    parameters = fhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(max(1, len(coeff_mod_bit_sizes) - 1))
    parameters.SetScalingModSize(global_scale_bits)
    parameters.SetRingDim(poly_modulus_degree)
    parameters.SetBatchSize(max(1, min(8, poly_modulus_degree // 2)))

    context = fhe.GenCryptoContext(parameters)
    context.Enable(fhe.PKESchemeFeature.PKE)
    context.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    context.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
    return context


def create_and_save_context(
    server_path: str,
    client_path: str,
    poly_modulus_degree: int = 16384,
    coeff_mod_bit_sizes: Optional[List[int]] = None,
    global_scale_bits: int = 29,
    generate_galois_keys: bool = False,
):
    """Generate CKKS context and save server/client variants."""
    fhe = _get_openfhe()

    context = _create_context(
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        global_scale_bits=global_scale_bits,
    )

    keypair = context.KeyGen()
    context.EvalMultKeyGen(keypair.secretKey)
    if generate_galois_keys:
        context.EvalRotateKeyGen(keypair.secretKey, [1, -1])

    _ensure_parent_dir(client_path)
    _ensure_parent_dir(server_path)

    if not fhe.SerializeToFile(client_path, context, fhe.BINARY):
        raise RuntimeError(f"Failed to serialize crypto context to {client_path}")
    if not fhe.SerializeToFile(f"{client_path}.sk", keypair.secretKey, fhe.BINARY):
        raise RuntimeError("Failed to serialize secret key")
    if not fhe.SerializeToFile(f"{client_path}.pk", keypair.publicKey, fhe.BINARY):
        raise RuntimeError("Failed to serialize public key")

    if not fhe.SerializeToFile(server_path, context, fhe.BINARY):
        raise RuntimeError(f"Failed to serialize crypto context to {server_path}")
    if not fhe.SerializeToFile(f"{server_path}.pk", keypair.publicKey, fhe.BINARY):
        raise RuntimeError("Failed to serialize public key")
    if not context.SerializeEvalMultKey(f"{server_path}.evalmult", fhe.BINARY):
        raise RuntimeError("Failed to serialize eval mult keys")
    if generate_galois_keys and not context.SerializeEvalAutomorphismKey(
        f"{server_path}.evalrot", fhe.BINARY
    ):
        raise RuntimeError("Failed to serialize eval rotation keys")

    return context, keypair


def load_context(path: str):
    """Load an OpenFHE context from disk."""
    fhe = _get_openfhe()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    context, success = fhe.DeserializeCryptoContext(path, fhe.BINARY)
    if not success:
        raise RuntimeError(f"Failed to deserialize crypto context from {path}")
    return context


def load_client_keys(path: str):
    """Load OpenFHE client keys and crypto context."""
    fhe = _get_openfhe()
    if not os.path.exists(f"{path}.sk") or not os.path.exists(f"{path}.pk"):
        raise FileNotFoundError(path)
    context = load_context(path)
    secret_key, success = fhe.DeserializePrivateKey(f"{path}.sk", fhe.BINARY)
    if not success:
        raise RuntimeError("Failed to deserialize private key")
    public_key, success = fhe.DeserializePublicKey(f"{path}.pk", fhe.BINARY)
    if not success:
        raise RuntimeError("Failed to deserialize public key")
    return context, secret_key, public_key


def load_server_keys(path: str):
    """Load OpenFHE server public key and eval keys for homomorphic ops."""
    fhe = _get_openfhe()
    if not os.path.exists(f"{path}.pk") or not os.path.exists(f"{path}.evalmult"):
        raise FileNotFoundError(path)
    context = load_context(path)
    public_key, success = fhe.DeserializePublicKey(f"{path}.pk", fhe.BINARY)
    if not success:
        raise RuntimeError("Failed to deserialize public key")
    if not context.DeserializeEvalMultKey(f"{path}.evalmult", fhe.BINARY):
        raise RuntimeError("Failed to deserialize eval mult keys")
    if os.path.exists(f"{path}.evalrot"):
        context.DeserializeEvalAutomorphismKey(f"{path}.evalrot", fhe.BINARY)
    return context, public_key


def serialize_ciphertext(ciphertext) -> bytes:
    """Serialize an OpenFHE ciphertext to bytes."""
    fhe = _get_openfhe()
    return bytes(fhe.Serialize(ciphertext, fhe.BINARY))


def deserialize_ciphertext(data: bytes):
    """Deserialize an OpenFHE ciphertext from bytes."""
    fhe = _get_openfhe()
    return fhe.DeserializeCiphertextString(data, fhe.BINARY)
