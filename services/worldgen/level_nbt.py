"""Minimal Java Edition NBT reader/writer for save metadata such as level.dat.

Only the standard library is used so the worldgen runner image needs nothing
beyond Python. Tag types are preserved exactly, so a file that is read and
written again without changes is byte-identical after decompression.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
from pathlib import Path
import struct
from typing import Any, BinaryIO


END, BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, BYTE_ARRAY, STRING, LIST, COMPOUND, INT_ARRAY, LONG_ARRAY = range(13)
SCALAR_FORMATS = {BYTE: ">b", SHORT: ">h", INT: ">i", LONG: ">q", FLOAT: ">f", DOUBLE: ">d"}
ARRAY_FORMATS = {BYTE_ARRAY: "b", INT_ARRAY: "i", LONG_ARRAY: "q"}


class NbtError(ValueError):
    """The payload is not well-formed NBT."""


@dataclass
class Tag:
    type: int
    value: Any
    element_type: int = END


def _read_exact(source: BinaryIO, size: int) -> bytes:
    data = source.read(size)
    if len(data) != size:
        raise NbtError("truncated NBT payload")
    return data


def _read_string(source: BinaryIO) -> str:
    (length,) = struct.unpack(">H", _read_exact(source, 2))
    return _read_exact(source, length).decode("utf-8")


def _write_string(target: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    target.write(struct.pack(">H", len(encoded)) + encoded)


def _read_payload(source: BinaryIO, tag_type: int, depth: int = 0) -> Tag:
    if depth > 512:
        raise NbtError("NBT nesting is too deep")
    if tag_type in SCALAR_FORMATS:
        fmt = SCALAR_FORMATS[tag_type]
        return Tag(tag_type, struct.unpack(fmt, _read_exact(source, struct.calcsize(fmt)))[0])
    if tag_type == STRING:
        return Tag(tag_type, _read_string(source))
    if tag_type in ARRAY_FORMATS:
        (length,) = struct.unpack(">i", _read_exact(source, 4))
        fmt = ARRAY_FORMATS[tag_type]
        if length < 0:
            raise NbtError("negative NBT array length")
        size = length * struct.calcsize(fmt)
        return Tag(tag_type, list(struct.unpack(f">{length}{fmt}", _read_exact(source, size))))
    if tag_type == LIST:
        element_type = _read_exact(source, 1)[0]
        (length,) = struct.unpack(">i", _read_exact(source, 4))
        if length < 0:
            raise NbtError("negative NBT list length")
        return Tag(tag_type, [_read_payload(source, element_type, depth + 1) for _ in range(length)], element_type)
    if tag_type == COMPOUND:
        entries: dict[str, Tag] = {}
        while (child_type := _read_exact(source, 1)[0]) != END:
            name = _read_string(source)
            entries[name] = _read_payload(source, child_type, depth + 1)
        return Tag(tag_type, entries)
    raise NbtError(f"unknown NBT tag type {tag_type}")


def _write_payload(target: BinaryIO, tag: Tag) -> None:
    if tag.type in SCALAR_FORMATS:
        target.write(struct.pack(SCALAR_FORMATS[tag.type], tag.value))
    elif tag.type == STRING:
        _write_string(target, tag.value)
    elif tag.type in ARRAY_FORMATS:
        fmt = ARRAY_FORMATS[tag.type]
        target.write(struct.pack(f">i{len(tag.value)}{fmt}", len(tag.value), *tag.value))
    elif tag.type == LIST:
        target.write(bytes([tag.element_type]) + struct.pack(">i", len(tag.value)))
        for element in tag.value:
            _write_payload(target, element)
    elif tag.type == COMPOUND:
        for name, child in tag.value.items():
            target.write(bytes([child.type]))
            _write_string(target, name)
            _write_payload(target, child)
        target.write(bytes([END]))
    else:
        raise NbtError(f"unknown NBT tag type {tag.type}")


def loads(payload: bytes) -> tuple[str, Tag]:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    source = io.BytesIO(payload)
    if _read_exact(source, 1)[0] != COMPOUND:
        raise NbtError("NBT root must be a compound")
    name = _read_string(source)
    root = _read_payload(source, COMPOUND)
    if source.read(1):
        raise NbtError("trailing bytes after NBT root")
    return name, root


def dumps(name: str, root: Tag, *, compress: bool = True) -> bytes:
    """Serialize as gzip-compressed NBT, the encoding used by level.dat and data/*.dat.

    ``compress=False`` returns the raw NBT that region files store per chunk.
    """
    target = io.BytesIO()
    target.write(bytes([COMPOUND]))
    _write_string(target, name)
    _write_payload(target, root)
    return gzip.compress(target.getvalue(), mtime=0) if compress else target.getvalue()


def load(path: Path) -> tuple[str, Tag]:
    return loads(path.read_bytes())


def save(path: Path, name: str, root: Tag) -> None:
    path.write_bytes(dumps(name, root))
