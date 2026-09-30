"""M3DS byte container in lossless RGB PNG pixels (copied from ComfyUI-YuE2Fast/m3ds.py; keep in sync).

Byte-compatible with UnlimitedEditing/Meshsmuggler (same header, gzip flag and CRC32s), so its
unsmuggle.html / UnsmuggleMeshFromImage also decode these images. Single chunk only here: direction sheets
are a few KB.
"""
import gzip
import io
import math
import struct
import zlib

import numpy as np
from PIL import Image

MAGIC = b"M3DS"
VERSION = 1
FLAG_GZIP = 0x01
HEADER_FMT = ">4sBBHHQIIQIH"  # magic, version, flags, chunk_index, chunk_count, blob_total,
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # chunk_payload_len, chunk_crc32, orig_len, orig_crc32, fname_len
CHANNELS = 3


def encode(data: bytes, filename: str, min_side: int = 256) -> Image.Image:
    blob = gzip.compress(data, compresslevel=9, mtime=0)
    fname = filename.encode("utf-8")
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, FLAG_GZIP, 0, 1, len(blob), len(blob),
                         zlib.crc32(blob) & 0xFFFFFFFF, len(data), zlib.crc32(data) & 0xFFFFFFFF, len(fname))
    raw = header + fname + blob
    pixels = math.ceil(len(raw) / CHANNELS)
    # Near-square like Meshsmuggler, but never tiny: clients are less tempted to rescale a 256px image.
    width = max(min_side, math.ceil(math.sqrt(pixels)))
    height = max(min_side, math.ceil(pixels / width))
    raw += b"\x00" * (width * height * CHANNELS - len(raw))  # decoders read lengths from the header
    return Image.fromarray(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, CHANNELS), "RGB")


def describe(image, size):
    return f"{image.format or '?'} {image.width}x{image.height} {image.mode}, {size} bytes"


def decode_png_bytes(png: bytes):
    image = Image.open(io.BytesIO(png))
    info = describe(image, len(png))
    if image.format != "PNG":
        raise ValueError(f"The sheet image arrived as {info}, not the original PNG: it was recompressed on the way "
                         "(chat apps convert photos to JPEG). Send the original PNG as a file/document.")
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    try:
        return decode_pixels(pixels.reshape(-1).tobytes())
    except ValueError as exc:
        # A host may have enlarged it with nearest-neighbour scaling: every source pixel becomes a
        # uniform k x k block, which is reversible. Smoothed or shrunk images are not.
        for k in range(2, 9):
            h, w = pixels.shape[0] // k * k, pixels.shape[1] // k * k
            if h < k or w < k:
                break
            blocks = pixels[:h, :w].reshape(h // k, k, w // k, k, 3)
            if (blocks == blocks[:, :1, :, :1]).all():
                try:
                    return decode_pixels(blocks[:, 0, :, 0].reshape(-1).tobytes())
                except ValueError:
                    pass
        raise ValueError(f"{exc} Received {info}.") from None


def decode_pixels(flat: bytes):
    if len(flat) < HEADER_SIZE:
        raise ValueError("Image too small to hold an M3DS header")
    (magic, version, flags, index, count, blob_total, payload_len, chunk_crc,
     orig_len, orig_crc, fname_len) = struct.unpack(HEADER_FMT, flat[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError("Not an M3DS image (bad magic): the pixels were changed after the job saved it "
                         "(recompressed, resized or re-encoded), or it isn't a Rig Studio direction sheet.")
    if version != VERSION or count != 1:
        raise ValueError(f"Unsupported M3DS image (version {version}, {count} chunks)")
    pos = HEADER_SIZE
    filename = flat[pos:pos + fname_len].decode("utf-8", "replace")
    payload = flat[pos + fname_len:pos + fname_len + payload_len]
    if len(payload) != payload_len or zlib.crc32(payload) & 0xFFFFFFFF != chunk_crc:
        raise ValueError("M3DS CRC mismatch: the PNG's pixels were altered (resized or lossy re-encode). Send it as a file.")
    data = gzip.decompress(payload) if flags & FLAG_GZIP else payload
    if len(data) != orig_len or zlib.crc32(data) & 0xFFFFFFFF != orig_crc:
        raise ValueError("M3DS final CRC mismatch: data corrupted")
    return data, filename
