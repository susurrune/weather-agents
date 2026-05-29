"""Self-signed TLS certificate generation for local network HTTPS.

Uses the ``cryptography`` library to generate a 2048-bit RSA key
and a self-signed X.509 certificate with Subject Alternative Names
(SANs) for all specified IP addresses.

Certificates are stored in ``~/.weather-agents/certs/`` and keyed
by a hash of the IP list so they are reused when addresses don't change.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import os
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_DIR = Path.home() / ".weather-agents" / "certs"


def detect_all_lan_ips() -> list[str]:
    """Return all non-loopback IPv4 addresses on this machine."""
    ips: list[str] = []
    try:
        hostname = socket.gethostname()
        ips = [ip for ip in socket.gethostbyname_ex(hostname)[2] if not ip.startswith("127.")]
    except Exception:
        pass

    # Also try UDP trick to get the default route interface IP.
    # ``socket.socket`` as a context manager guarantees close on every
    # path; the previous explicit ``s.close()`` leaked the fd when
    # ``connect`` raised (e.g. offline machine).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.2)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        if ip not in ips:
            ips.append(ip)
    except Exception:
        pass

    return ips


def _cert_key_paths(ips: list[str]) -> tuple[Path, Path]:
    """Return (cert_path, key_path) for the given IP list.

    Uses a hash of sorted IPs so changes in the IP set produce a new
    cert while stable IPs reuse the existing one.
    """
    h = hashlib.sha256(",".join(sorted(ips)).encode()).hexdigest()[:12]
    return CERT_DIR / f"voice_{h}.pem", CERT_DIR / f"voice_{h}.key"


def ensure_self_signed_cert(ips: list[str]) -> tuple[str, str]:
    """Idempotently ensure a self-signed cert+key for *ips* exists.

    Returns ``(cert_path, key_path)`` strings.
    If valid files already exist for this exact IP set they are returned as-is.
    """
    if not ips:
        ips = ["127.0.0.1"]

    cert_path, key_path = _cert_key_paths(ips)
    if cert_path.is_file() and key_path.is_file():
        return str(cert_path), str(key_path)

    CERT_DIR.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Save private key
    key_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path.write_bytes(key_bytes)
    os.chmod(key_path, 0o600)

    # Build self-signed certificate with SAN for each IP
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Weather Agents"),
            x509.NameAttribute(NameOID.COMMON_NAME, "voice-server"),
        ]
    )

    san = x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips])

    # cryptography>=42 deprecates naive datetimes for not_valid_before/after;
    # use timezone-aware UTC. ``timedelta`` avoids the Feb-29 edge case where
    # ``now.replace(year=now.year + 10)`` would raise on a non-leap target.
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365 * 10))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    return str(cert_path), str(key_path)
