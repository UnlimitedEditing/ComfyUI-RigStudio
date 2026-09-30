"""Resolve Graydient media inputs and tell a direction sheet from audio by its bytes.

A Graydient field can arrive as a real URL, a staged filename in input/, or a delimiter-stripped
Telegram reference (catalog KI-007 §7-8). The Direct and Perform nodes take several such inputs and
sort them by content: a PNG is a direction sheet, anything else is audio. The resolver pattern is
ported from ComfyUI-YuE2Fast/cover.py (itself from ComfyUI-HiggsV3Glue, confirmed live).
"""
import asyncio
import hashlib
import os
import re
import tempfile
import urllib.parse
import urllib.request

_MANGLED_TELEGRAM = re.compile(
    r'^(?:[a-z_]+__)?(https?)api\.telegram\.orgfilebot(\d+)([A-Za-z0-9_-]+?)'
    r'(voice|photo|video_note|video|audio|document|animation|sticker)file(\d+)\.([a-z0-9]+)$',
    re.IGNORECASE,
)
PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def unmangle_telegram(value):
    m = _MANGLED_TELEGRAM.match(value)
    if not m:
        return None
    scheme, bot_id, secret, kind, file_id, ext = m.groups()
    return f"{scheme}://api.telegram.org/file/bot{bot_id}:{secret}/{kind}/file_{file_id}.{ext}"


def _download(url):
    suffix = os.path.splitext(url.split("?")[0])[1][:8] or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="rigstudio-")
    os.close(fd)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # some WAFs 403 urllib's UA
    with urllib.request.urlopen(request, timeout=180) as response, open(path, "wb") as f:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            f.write(block)
    return path


async def resolve(value, label, log=print):
    """-> (local path, public URL or ""). URL / Telegram reference / file in input/."""
    value = value.strip()
    if value.startswith(("http://", "https://")):
        path = await asyncio.to_thread(_download, value)
        log(f"{label}: downloaded from {urllib.parse.urlsplit(value).hostname}")
        return path, value
    url = unmangle_telegram(value)
    if url:
        log(f"{label}: Telegram reference, reconstructed")
        # A bot-token URL is not a durable public address: never store it in a sheet.
        return await asyncio.to_thread(_download, url), ""
    import folder_paths
    for candidate in (value, os.path.join(folder_paths.get_input_directory(), value)):
        if os.path.isfile(candidate):
            log(f"{label}: local file {os.path.basename(candidate)}")
            return candidate, ""
    raise ValueError(f"{label}={value!r} is not an http(s) URL, a Telegram file reference, or a file in input/")


def is_png(path):
    with open(path, "rb") as f:
        return f.read(8) == PNG_MAGIC


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


async def sort_inputs(values, log=print):
    """{label: value} -> (sheet (path, url) or None, audio (path, url) or None). First of each kind wins."""
    sheet = audio = None
    seen = set()
    for label, value in values.items():
        if not (value or "").strip() or value.strip() in seen:
            continue
        seen.add(value.strip())
        path, url = await resolve(value, label, log)
        if is_png(path):
            if sheet is None:
                sheet = (path, url)
                log(f"{label}: direction sheet PNG")
        elif audio is None:
            audio = (path, url)
            log(f"{label}: audio")
    return sheet, audio
