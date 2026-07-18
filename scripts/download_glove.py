"""Download a single official GloVe vector member with resumable HTTP ranges."""

from __future__ import annotations

import argparse
import struct
import urllib.request
import zlib
from pathlib import Path

from entitylens.config import Paths

GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(50, 100, 200, 300), default=50)
    return parser.parse_args()


def fetch_range(start: int, end: int) -> tuple[bytes, str]:
    """Fetch one byte range while rejecting servers that ignore range requests."""
    request = urllib.request.Request(GLOVE_URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith("bytes "):
            raise RuntimeError("The GloVe host did not honour the requested byte range")
        return response.read(), content_range


def append_range_to_file(start: int, end: int, output: Path) -> None:
    """Stream a byte range to disk so interrupted transfers remain resumable."""
    request = urllib.request.Request(GLOVE_URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError("The GloVe host did not honour the requested byte range")
        with output.open("ab") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)


def archive_size() -> int:
    """Read the total archive size from a one-byte range response."""
    _, content_range = fetch_range(0, 0)
    return int(content_range.rsplit("/", maxsplit=1)[1])


def member_details(filename: str) -> tuple[int, int, int]:
    """Locate a ZIP member's compression method, payload offset, and compressed size."""
    total_size = archive_size()
    tail, _ = fetch_range(max(0, total_size - 65_557), total_size - 1)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0:
        raise RuntimeError("Could not find the GloVe ZIP directory")
    _, _, _, _, _, directory_size, directory_offset, _ = struct.unpack_from(
        "<4s4H2IH", tail, marker
    )
    directory, _ = fetch_range(directory_offset, directory_offset + directory_size - 1)
    offset = 0
    while offset < len(directory):
        fields = struct.unpack_from("<4s6H3I5H2I", directory, offset)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError("Malformed GloVe ZIP directory entry")
        name_length, extra_length, comment_length = fields[10:13]
        name_start = offset + 46
        name = directory[name_start : name_start + name_length].decode("utf-8")
        if name == filename:
            local_offset = fields[16]
            header, _ = fetch_range(local_offset, local_offset + 29)
            local = struct.unpack("<4s5H3I2H", header)
            if local[0] != b"PK\x03\x04":
                raise RuntimeError("Malformed GloVe ZIP local header")
            payload_offset = local_offset + 30 + local[9] + local[10]
            return fields[4], payload_offset, fields[8]
        offset = name_start + name_length + extra_length + comment_length
    raise FileNotFoundError(f"{filename} is not in the official GloVe archive")


def write_member(destination: Path, filename: str) -> None:
    """Resume a compressed member transfer, then inflate it into its text file."""
    output = destination / filename
    if output.is_file():
        print(f"using existing {output}")
        return
    method, payload_offset, compressed_size = member_details(filename)
    partial = destination / f"{filename}.partial.v2"
    downloaded = partial.stat().st_size if partial.is_file() else 0
    if downloaded < compressed_size:
        append_range_to_file(
            payload_offset + downloaded,
            payload_offset + compressed_size - 1,
            partial,
        )
        downloaded = partial.stat().st_size
    if downloaded < compressed_size:
        print(f"resumable download: {downloaded:,} / {compressed_size:,} bytes")
        return
    compressed = partial.read_bytes()
    if method == 8:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif method == 0:
        data = compressed
    else:
        raise RuntimeError(f"Unsupported GloVe ZIP compression method: {method}")
    output.write_bytes(data)
    partial.unlink(missing_ok=True)
    print(f"wrote {output}")


def main() -> None:
    args = parse_args()
    destination = Paths().root / "data" / "embeddings"
    destination.mkdir(parents=True, exist_ok=True)
    write_member(destination, f"glove.6B.{args.dimension}d.txt")


if __name__ == "__main__":
    main()
