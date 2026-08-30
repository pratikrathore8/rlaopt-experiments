"""Independent, deterministic random streams."""

import hashlib


def derive_seed(master: int, stream: str) -> int:
    payload = f"rlaopt-experiments-v1:{master}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)
