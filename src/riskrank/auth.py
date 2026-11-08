"""Dashboard login (NFR15): one user name and password; only a salted hash is stored, outside the
repository.

    python -m riskrank.auth ~/.config/riskrank/dashboard-auth.json      # set or replace the login

The file holds {"user", "salt", "hash", "n", "r", "p"}: scrypt (RFC 7914) over the password with a
random 16-byte salt. It is written with owner-only permissions.
"""

from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys

N, R, P = 2**14, 8, 1


def _hash(password: str, salt: bytes, n: int = N, r: int = R, p: int = P) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)


def make_record(user: str, password: str) -> dict:
    salt = secrets.token_bytes(16)
    return {"user": user, "salt": salt.hex(), "hash": _hash(password, salt).hex(), "n": N, "r": R, "p": P}


def verify(record: dict, user: str, password: str) -> bool:
    expected = bytes.fromhex(record["hash"])
    got = _hash(password, bytes.fromhex(record["salt"]), record["n"], record["r"], record["p"])
    # compare both parts in constant time, and always do the hash so a wrong user takes as long
    return hmac.compare_digest(user.encode(), record["user"].encode()) & hmac.compare_digest(got, expected)


def load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main() -> None:
    path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.config/riskrank/dashboard-auth.json")
    user = input("Dashboard user name: ").strip()
    pw = getpass.getpass("Password: ")
    if len(pw) < 12 or pw != getpass.getpass("Repeat: "):
        sys.exit("Passwords differ or are shorter than 12 characters.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(make_record(user, pw), f)
    print(f"Login for {user!r} written to {path}")


if __name__ == "__main__":
    main()
