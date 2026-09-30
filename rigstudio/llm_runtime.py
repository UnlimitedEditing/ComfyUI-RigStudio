"""Private Ollama for the Direct node: the exact model + engine the director was benchmarked with.

Director v3 scored 71% on the approved benchmark with Ollama's library `qwen3:14b` (Q4_K_M GGUF,
Ollama's own chat template, think=False) on Ollama 0.34.2. Rather than approximate that with a
different runtime/quantisation, the job runs the same thing:

  * the official Ollama 0.34.2 linux-amd64 build (GitHub release .tar.zst) — concept-staged to
    models/rigstudio/ollama/ when possible, else downloaded here; unpacked with `zstandard` from a
    private `pip --target` dir (catalog KI-007 §13: nothing installed into ComfyUI's venv);
  * the model's registry blobs concept-staged to models/ollama/blobs/ (sha256-<digest>), exposed
    through a private OLLAMA_MODELS dir of symlinks plus the manifest written below. Missing blobs
    fall back to `ollama pull` (inside the job budget — slow, but works);
  * `ollama serve` as an async child in its own process group, killed when the node finishes.
"""
import asyncio
import json
import os
import shutil
import signal
import sys
import time
import urllib.request

from . import runtime as R

OLLAMA_VERSION = "0.34.2"
OLLAMA_URL = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
OLLAMA_STAGED = f"rigstudio/ollama/ollama-linux-amd64-v{OLLAMA_VERSION}.tar.zst"   # under models/
BLOBS_STAGED = "ollama/blobs"                                                     # under models/
REGISTRY = "https://registry.ollama.ai/v2/library/"
ZSTD_PKG = "zstandard==0.23.0"
SKIP_LIBS = ("cuda_v13", "rocm", "vulkan", "mlx")   # keep bin/, the CPU backends and cuda_v12

# registry.ollama.ai/library/<name>/<tag> manifests, copied from the local pulls the benchmark used
MANIFESTS = {
    "qwen3:14b": {"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                  "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                             "digest": "sha256:78b3b822087d5199783c8203553a5a92ce5eb7b683a5a81003f8efea9b399d74",
                             "size": 488},
                  "layers": [
                      {"mediaType": "application/vnd.ollama.image.model",
                       "digest": "sha256:a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e",
                       "size": 9276184896},
                      {"mediaType": "application/vnd.ollama.image.template",
                       "digest": "sha256:ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4", "size": 1723},
                      {"mediaType": "application/vnd.ollama.image.license",
                       "digest": "sha256:d18a5cc71b84bc4af394a31116bd3932b42241de70c77d2b76d69a314ec8aa12", "size": 11338},
                      {"mediaType": "application/vnd.ollama.image.params",
                       "digest": "sha256:cff3f395ef3756ab63e58b0ad1b32bb6f802905cae1472e6a12034e4246fbbdb", "size": 120}]},
}

_UNPACK = r"""
import sys, tarfile, zstandard
src, dest, skip = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
n = 0
with open(src, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as z, tarfile.open(fileobj=z, mode="r|") as tf:
    for m in tf:
        if any(s in m.name for s in skip):
            continue
        tf.extract(m, dest)
        n += 1
print(n, "members")
"""


def blob_urls(model):
    """(url, destination under models/) for concept_mapping — used by the gen script."""
    name = model.split(":")[0]
    m = MANIFESTS[model]
    return [(f"{REGISTRY}{name}/blobs/{d['digest']}", f"{BLOBS_STAGED}/{d['digest'].replace(':', '-')}")
            for d in [m["config"], *m["layers"]]]


async def ensure_ollama(log):
    """-> path of the ollama binary (unpacked once per container)."""
    import folder_paths
    root = os.path.join(R.WORK_DIR, f"ollama-{OLLAMA_VERSION}")
    exe = os.path.join(root, "bin", "ollama")
    if os.path.isfile(exe):
        return exe
    t0 = time.time()
    staged = os.path.join(folder_paths.models_dir, OLLAMA_STAGED)
    if os.path.isfile(staged):
        src, how = staged, "staged"
    else:
        src, how = os.path.join(R.WORK_DIR, os.path.basename(OLLAMA_STAGED)), "downloaded"
        await R._download(OLLAMA_URL, src)
    t_get = time.time() - t0
    priv = os.path.join(R.WORK_DIR, "zstd-deps")
    if not os.path.isdir(os.path.join(priv, "zstandard")):
        rc, _, err = await R.run_subprocess([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                                             "--target", priv, ZSTD_PKG], stream_prefix="pip-zstd")
        if rc != 0:
            raise RuntimeError(f"zstandard install failed (rc={rc}):\n{err[-2000:]}")
    tmp = root + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    rc, out, err = await R.run_subprocess([sys.executable, "-c", _UNPACK, src, tmp, ",".join(SKIP_LIBS)],
                                          env={"PYTHONPATH": priv})
    if rc != 0:
        raise RuntimeError(f"ollama unpack failed (rc={rc}):\n{err[-2000:]}")
    os.replace(tmp, root)
    os.chmod(exe, 0o755)
    log(f"ollama {OLLAMA_VERSION} {how} in {t_get:.1f}s, unpacked {out.strip()} in {time.time() - t0 - t_get:.1f}s")
    return exe


def prepare_models(model, log):
    """Private OLLAMA_MODELS dir: symlinks to concept-staged blobs + the manifest.
    -> (models dir, number of blobs still missing)."""
    import folder_paths
    home = os.path.join(R.WORK_DIR, "ollama-models")
    name, tag = model.split(":")
    blobs = os.path.join(home, "blobs")
    os.makedirs(blobs, exist_ok=True)
    staged = os.path.join(folder_paths.models_dir, BLOBS_STAGED)
    m = MANIFESTS[model]
    missing = 0
    for d in [m["config"], *m["layers"]]:
        fname = d["digest"].replace(":", "-")
        src, dst = os.path.join(staged, fname), os.path.join(blobs, fname)
        if os.path.isfile(src) and os.path.getsize(src) == d["size"]:
            # re-point every run: on a warm machine an old link can dangle into a deleted project dir
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
        elif os.path.isfile(dst) and not os.path.islink(dst):
            continue                                   # pulled in-job on an earlier run
        else:
            if os.path.lexists(dst):
                os.remove(dst)
            missing += 1
    if not missing:
        mdir = os.path.join(home, "manifests", "registry.ollama.ai", "library", name)
        os.makedirs(mdir, exist_ok=True)
        with open(os.path.join(mdir, tag), "w") as f:
            json.dump(m, f, separators=(",", ":"))
    log(f"ollama model {model}: {'all blobs staged' if not missing else f'{missing} blob(s) not staged, will pull'}")
    return home, missing


class OllamaServer:
    """async with OllamaServer(exe, models_dir) as url: ...  (port picked free, child killed on exit)"""

    def __init__(self, exe, models_dir, log=print):
        self.exe, self.models, self.log, self.proc = exe, models_dir, log, None

    async def __aenter__(self):
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_MODELS": self.models,
               "OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_NOPRUNE": "1"}
        logf = open(os.path.join(R.WORK_DIR, "ollama-serve.log"), "ab")
        self.proc = await asyncio.create_subprocess_exec(self.exe, "serve", stdout=logf, stderr=logf, env=env,
                                                         start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 60:
            if self.proc.returncode is not None:
                raise RuntimeError(f"ollama serve exited rc={self.proc.returncode}: {self.tail()}")
            try:
                await asyncio.to_thread(urllib.request.urlopen, self.url + "/api/version", None, 2)
                self.log(f"ollama serve up in {time.time() - t0:.1f}s")
                return self.url
            except Exception:
                await asyncio.sleep(0.5)
        raise RuntimeError(f"ollama serve did not come up in 60 s: {self.tail()}")

    def tail(self, n=3000):
        try:
            with open(os.path.join(R.WORK_DIR, "ollama-serve.log"), "rb") as f:
                return f.read()[-n:].decode(errors="replace")
        except OSError:
            return ""

    async def __aexit__(self, *exc):
        if self.proc and self.proc.returncode is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                await asyncio.wait_for(self.proc.wait(), 10)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


async def pull(url, model, log):
    t0 = time.time()
    body = json.dumps({"model": model, "stream": False}).encode()
    req = urllib.request.Request(url + "/api/pull", data=body, headers={"Content-Type": "application/json"})
    await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=1800).read())
    log(f"pulled {model} in {time.time() - t0:.1f}s")
