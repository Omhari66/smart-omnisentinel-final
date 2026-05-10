"""
scripts/generate_selfsigned_cert.py
------------------------------------
Generates a self-signed TLS certificate for local / internal-network HTTPS.
Uses the `cryptography` package (already a project dependency).

Usage:
    python scripts/generate_selfsigned_cert.py

Output:
    deployments/nginx/certs/selfsigned.key  — RSA 2048-bit private key (PEM)
    deployments/nginx/certs/selfsigned.crt  — X.509 self-signed certificate (PEM)

The certificate is valid for 365 days and covers:
  - CN = smartomnisentinel
  - SAN: localhost, 127.0.0.1 (browser SAN requirement)
"""

from __future__ import annotations

import datetime
import ipaddress
import os
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

# ── Output paths ──────────────────────────────────────────────────────────────
CERTS_DIR = Path(__file__).parent.parent / "deployments" / "nginx" / "certs"
KEY_PATH  = CERTS_DIR / "selfsigned.key"
CERT_PATH = CERTS_DIR / "selfsigned.crt"

CERTS_DIR.mkdir(parents=True, exist_ok=True)


def generate() -> None:
    print("Generating RSA 2048-bit private key...")
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Write private key (no passphrase — nginx reads it without interaction)
    KEY_PATH.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    print(f"  Private key → {KEY_PATH}")

    # ── Certificate subject / issuer ─────────────────────────────────────────
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "India"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SmartOmniSentinel"),
        x509.NameAttribute(NameOID.COMMON_NAME, "smartomnisentinel"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)

    print("Generating self-signed certificate (365 days)...")
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        # Subject Alternative Names — browsers REQUIRE these (CN alone is ignored)
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("smartomnisentinel"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        # Mark as CA: false (it's a leaf cert, not a CA)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"  Certificate  → {CERT_PATH}")

    print()
    print("✅ Done! Files created:")
    print(f"   {KEY_PATH}")
    print(f"   {CERT_PATH}")
    print()
    print("Next steps:")
    print("  1. These files are gitignored (private key must never be committed).")
    print("  2. Nginx will use them for HTTPS on port 443.")
    print("  3. Your browser will show a security warning — click 'Advanced → Proceed'.")
    print("     This is expected for self-signed certs.")


if __name__ == "__main__":
    generate()
