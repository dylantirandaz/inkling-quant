"""Build the pinned WikiText test corpus for the Modal measurement image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
import zipfile
from contextlib import suppress
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Final

SCHEMA_VERSION: Final = "inkling-wikitext2-raw-test-reference-v2"
REFERENCE_HASH_DOMAIN: Final = b"inkling-wikitext2-raw-test-reference-v2\0"
MAX_ARCHIVE_BYTES: Final = 8 * 1024 * 1024
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


def _strict_object(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate reference key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite reference value: {value}")

    raw_bytes = path.read_bytes()
    value = json.loads(
        raw_bytes.decode("utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )
    if type(value) is not dict:
        raise ValueError("corpus reference must be one JSON object")
    canonical = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw_bytes != canonical:
        raise ValueError("corpus reference must use canonical JSON plus one newline")
    reference_hash = value.get("reference_sha256")
    if type(reference_hash) is not str or SHA256_RE.fullmatch(reference_hash) is None:
        raise ValueError("corpus reference self-hash must be a lowercase SHA-256")
    payload = dict(value)
    del payload["reference_sha256"]
    payload_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(REFERENCE_HASH_DOMAIN + payload_bytes).hexdigest() != reference_hash:
        raise ValueError("corpus reference self-hash differs from its payload")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _positive_int(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{label} must be a bounded positive integer")
    return value


def _expected_hash(value: object, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _download(reference: dict[str, Any]) -> bytes:
    url = reference.get("archive_url")
    if type(url) is not str or not url.startswith("https://"):
        raise ValueError("archive URL must use HTTPS")
    expected_size = _positive_int(
        reference.get("archive_size_bytes"),
        label="archive size",
        maximum=MAX_ARCHIVE_BYTES,
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read(expected_size + 1)
    if not isinstance(archive, bytes):
        raise TypeError("archive response must contain bytes")
    if len(archive) != expected_size:
        raise ValueError("downloaded archive size differs from the reference")
    if _sha256(archive) != _expected_hash(
        reference.get("archive_sha256"),
        label="archive hash",
    ):
        raise ValueError("downloaded archive hash differs from the reference")
    return archive


def _extract(reference: dict[str, Any], archive: bytes) -> bytes:
    member = _canonical_relative_path(reference.get("archive_member"), label="archive member")
    with zipfile.ZipFile(BytesIO(archive)) as bundle:
        matches = tuple(info for info in bundle.infolist() if info.filename == member)
        if len(matches) != 1 or matches[0].is_dir():
            raise ValueError("archive does not contain one exact regular corpus member")
        corpus = bundle.read(matches[0])
    expected_size = _positive_int(
        reference.get("corpus_size_bytes"),
        label="corpus size",
        maximum=4 * 1024 * 1024,
    )
    if len(corpus) != expected_size:
        raise ValueError("extracted corpus size differs from the reference")
    if _sha256(corpus) != _expected_hash(
        reference.get("corpus_sha256"),
        label="corpus hash",
    ):
        raise ValueError("extracted corpus hash differs from the reference")
    return corpus


def materialize(reference_path: Path, output_path: Path) -> None:
    reference = _strict_object(reference_path)
    if reference.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported corpus reference schema")
    materializer = _canonical_relative_path(
        reference.get("materializer_path"),
        label="materializer path",
    )
    if not materializer.endswith("/materialize_inkling_measurement_corpus.py"):
        raise ValueError("corpus reference names a different materializer")
    if _sha256(Path(__file__).read_bytes()) != _expected_hash(
        reference.get("materializer_sha256"),
        label="materializer hash",
    ):
        raise ValueError("materializer source hash differs from the reference")
    expected_output = reference.get("materialized_path")
    if type(expected_output) is not str or "\\" in expected_output or "\x00" in expected_output:
        raise ValueError("materialized path must be one canonical absolute POSIX path")
    posix_output = PurePosixPath(expected_output)
    if (
        not posix_output.is_absolute()
        or expected_output.startswith("//")
        or posix_output.as_posix() != expected_output
        or any(part in {"", ".", ".."} for part in posix_output.parts[1:])
    ):
        raise ValueError("materialized path must be one canonical absolute POSIX path")
    if output_path.as_posix() != expected_output:
        raise ValueError("requested output path differs from the corpus reference")

    corpus = _extract(reference, _download(reference))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o444,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(corpus)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, output_path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    if output_path.read_bytes() != corpus:
        raise RuntimeError("materialized corpus failed exact readback")
    print(
        json.dumps(
            {
                "corpus_sha256": _sha256(corpus),
                "corpus_size_bytes": len(corpus),
                "materialized_path": output_path.as_posix(),
                "schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    materialize(arguments.reference.resolve(), arguments.output)


if __name__ == "__main__":
    main()
