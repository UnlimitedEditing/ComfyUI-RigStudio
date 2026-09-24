"""Isolated Audio2Face-3D runtime for ComfyUI nodes (Rig Studio).

Nothing here touches ComfyUI's own Python environment:
  * the prebuilt rigstudio-a2f bundle (binary + NVIDIA SDK libaudio2x.so) comes from this
    repo's GitHub release,
  * TensorRT 10.13.3.9 and the CUDA 12 runtime/cuBLAS/cuRAND are pip-installed with
    --target into a private folder (installing them into ComfyUI's venv could fight
    torch's pinned nvidia-* wheels),
  * every child process is async (a synchronous subprocess.run() inside a node blocks
    ComfyUI's event loop; on Graydient that made finished jobs report timed_out).
Everything is cached under WORK_DIR, so a warm container reuses it.
"""
import asyncio
import glob
import re
import os
import signal
import sys
import tarfile
import tempfile
import time
import urllib.request

WORK_DIR = os.path.join(tempfile.gettempdir(), "rigstudio")
RELEASE = "https://github.com/UnlimitedEditing/ComfyUI-RigStudio/releases/download/a2f-runtime-v1"
BUNDLE = "rigstudio-a2f-linux-x86_64-cuda12-trt10.13.tar.gz"
# Ampere+ hardware-compatible engine for Audio2Face-3D v2.3 Mark (NVIDIA Open Model License, see the
# release's NOTICE). Default for RigStudioBuildTrack: Graydient's option parser splits slot values on
# "/", so URLs cannot be passed through slots — the node falls back to this instead.
DEFAULT_ENGINE_URL = ("https://github.com/UnlimitedEditing/ComfyUI-RigStudio/releases/download/a2f-engine-v1/"
                      "audio2face-3d-v2.3-mark-trt10.13.3-ampere_plus.trt")
PIP_PKGS = [
    "tensorrt-cu12-libs==10.13.3.9",
    "tensorrt-cu12-bindings==10.13.3.9",
    "nvidia-cuda-runtime-cu12==12.9.79",
    "nvidia-cublas-cu12==12.9.1.4",
    "nvidia-curand-cu12==10.3.10.19",
]
HF_MARK = "https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark/resolve/main/"
MARK_FILES = ["model.json", "network.onnx", "network_info.json", "model_config.json", "model_data.npz",
              "implicit_emo_db.npz", "bs_skin.npz", "bs_skin_config.json", "bs_tongue.npz",
              "bs_tongue_config.json", "trt_info.json"]


async def run_subprocess(args, stream_prefix=None, env=None):
    """Async child in its own process group, killed afterwards; optional line echo with timestamps."""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True, limit=1 << 20, env={**os.environ, **env} if env else None)
    out_lines = []
    try:
        stderr_task = asyncio.create_task(proc.stderr.read())
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="replace").rstrip("\n")
            out_lines.append(line)
            if stream_prefix:
                print(f"[{stream_prefix} {time.strftime('%H:%M:%S')}] {line}", flush=True)
        stderr = await stderr_task
        await proc.wait()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL if proc.returncode is None else signal.SIGTERM)
        except ProcessLookupError:
            pass
    return proc.returncode, "\n".join(out_lines), stderr.decode(errors="replace")


async def _download(url, dest):
    if not os.path.isfile(dest):
        tmp = dest + ".part"
        await asyncio.to_thread(urllib.request.urlretrieve, url, tmp)
        os.replace(tmp, dest)
    return dest


async def ensure_runtime(log):
    """-> dict(bin, lib_path_env, python_path, timings). Provisions bundle + private pip deps once."""
    os.makedirs(WORK_DIR, exist_ok=True)
    info = {}
    t0 = time.time()
    root = os.path.join(WORK_DIR, "rigstudio-a2f")
    if not os.path.isfile(os.path.join(root, "bin", "rigstudio-a2f")):
        tar = await _download(f"{RELEASE}/{BUNDLE}", os.path.join(WORK_DIR, BUNDLE))
        with tarfile.open(tar) as tf:
            tf.extractall(WORK_DIR)
        os.chmod(os.path.join(root, "bin", "rigstudio-a2f"), 0o755)
    info["bundle_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    priv = os.path.join(WORK_DIR, "pydeps")
    marker = os.path.join(priv, ".ok-" + "-".join(p.split("==")[1] for p in PIP_PKGS))
    if not os.path.isfile(marker):
        rc, out, err = await run_subprocess(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", priv,
             "--extra-index-url", "https://pypi.nvidia.com", *PIP_PKGS], stream_prefix="pip")
        if rc != 0:
            raise RuntimeError(f"private pip install failed (rc={rc}):\n{err[-3000:]}")
        open(marker, "w").close()
    info["pydeps_s"] = round(time.time() - t0, 2)

    lib_dirs = [os.path.join(root, "lib"), os.path.join(priv, "tensorrt_libs")]
    lib_dirs += sorted(glob.glob(os.path.join(priv, "nvidia", "*", "lib")))
    env = {"LD_LIBRARY_PATH": ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]),
           "PYTHONPATH": priv}
    log(f"a2f runtime ready: bundle {info['bundle_s']}s, pydeps {info['pydeps_s']}s")
    return {"root": root, "bin": os.path.join(root, "bin", "rigstudio-a2f"), "env": env, "timings": info}


async def ensure_mark_model(log):
    """Mark v2.3 model files: concept_mapping stages them under models/audio2face/mark-v2.3/;
    falls back to downloading from NVIDIA's (ungated) HF repo if they are missing."""
    try:
        import folder_paths
        staged = os.path.join(folder_paths.models_dir, "audio2face", "mark-v2.3")
    except ImportError:
        staged = None
    t0 = time.time()
    if staged and all(os.path.isfile(os.path.join(staged, f)) for f in MARK_FILES):
        src, how = staged, "concept_mapping"
    else:
        src, how = os.path.join(WORK_DIR, "mark-v2.3"), "hf-download"
        os.makedirs(src, exist_ok=True)
        await asyncio.gather(*(_download(HF_MARK + f, os.path.join(src, f)) for f in MARK_FILES))
    log(f"mark model from {how} in {time.time() - t0:.1f}s")
    return src, how, round(time.time() - t0, 2)


async def build_engine(rt, onnx, trt_info, out_path, hardware_compat, log):
    """ONNX -> TensorRT engine via the private TensorRT Python bindings (trtexec is not in pip)."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trt_build.py")
    t0 = time.time()
    rc, out, err = await run_subprocess(
        [sys.executable, script, onnx, trt_info, out_path, "1" if hardware_compat else "0"],
        stream_prefix="trt", env=rt["env"])
    if rc != 0 or not os.path.isfile(out_path):
        raise RuntimeError(f"engine build failed (rc={rc}):\n{out[-2000:]}\n{err[-3000:]}")
    secs = round(time.time() - t0, 2)
    log(f"engine built in {secs}s ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return secs


def model_workdir(src, engine_path, name):
    """A model.json directory whose network.trt is our engine (support files symlinked)."""
    wd = os.path.join(WORK_DIR, "models", name)
    os.makedirs(wd, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith((".json", ".npz")):
            dst = os.path.join(wd, f)
            if not os.path.exists(dst):
                os.symlink(os.path.join(src, f), dst)
    trt = os.path.join(wd, "network.trt")
    if os.path.lexists(trt):
        os.remove(trt)
    os.symlink(engine_path, trt)
    return os.path.join(wd, "model.json")


ANALYSIS_PKGS = ["librosa>=0.10.2,<2", "soundfile"]


async def ensure_analysis(log):
    """librosa & co. for track_job.py, installed WITH deps into their own private folder — the
    analysis runs as a subprocess on that PYTHONPATH, so ComfyUI's numpy/scipy are never touched."""
    priv = os.path.join(WORK_DIR, "analysis-deps")
    marker = os.path.join(priv, ".ok-" + "-".join(re.sub(r"[^\w.]", "", p) for p in ANALYSIS_PKGS))
    t0 = time.time()
    if not os.path.isfile(marker):
        rc, out, err = await run_subprocess(
            [sys.executable, "-m", "pip", "install", "--quiet", "--target", priv, *ANALYSIS_PKGS],
            stream_prefix="pip-analysis")
        if rc != 0:
            raise RuntimeError(f"analysis deps install failed (rc={rc}):\n{err[-3000:]}")
        open(marker, "w").close()
    log(f"analysis deps ready in {time.time() - t0:.1f}s")
    return {"PYTHONPATH": priv}, round(time.time() - t0, 2)


async def fetch_audio_16k(src, log):
    """URL or local path -> 16 kHz mono wav via the system ffmpeg (torchaudio/torchcodec are
    unreliable on Graydient). Returns (wav_path, duration_s)."""
    d = os.path.join(WORK_DIR, "audio")
    os.makedirs(d, exist_ok=True)
    raw = src
    if re.match(r"^https?://", src):
        raw = os.path.join(d, "input" + (os.path.splitext(src.split("?")[0])[1] or ".bin"))
        if os.path.exists(raw):
            os.remove(raw)
        await asyncio.to_thread(urllib.request.urlretrieve, src, raw)
    wav = os.path.join(d, "input16k.wav")
    rc, _, err = await run_subprocess(["ffmpeg", "-v", "error", "-y", "-i", raw, "-ac", "1", "-ar", "16000",
                                       "-c:a", "pcm_s16le", wav])
    if rc != 0:
        raise RuntimeError(f"ffmpeg could not decode {src}: {err[-800:]}")
    duration = os.path.getsize(wav) / 32000.0     # 16 kHz * 2 bytes, header negligible
    log(f"audio ready: {duration:.1f}s")
    return wav, duration


async def fetch_engine(url, log):
    """A prebuilt (Ampere+ hardware-compatible) engine by URL (or local path), cached."""
    if os.path.isfile(url):
        return url
    d = os.path.join(WORK_DIR, "engines")
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, "downloaded-" + re.sub(r"[^\w.-]", "_", url.split("/")[-1].split("?")[0]))
    t0 = time.time()
    await _download(url, dest)
    log(f"engine downloaded in {time.time() - t0:.1f}s ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest
