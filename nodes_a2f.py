"""Rig Studio Audio2Face-3D nodes.

RigStudioA2FProbe — one-shot confirmation job: provisions the isolated A2F runtime, builds a
TensorRT engine for Mark v2.3 (optionally Ampere+ hardware-compatible, so it can be reused on
every newer GPU), runs A2F on the SDK's 4 s sample and reports every timing + environment
fact as a JSON data image. Its engine_path output can be smuggled out with Meshsmuggler.

Audio2Emotion is never used (licence: only inside NVIDIA's A2E->A2F pipeline, and not for
emotion recognition).
"""
import asyncio
import json
import math
import os
import platform
import struct
import sys
import time

import numpy as np
import torch

from .rigstudio import runtime as R


def encode_string_as_image(text):
    """UTF-8 -> RGB IMAGE: 4-byte big-endian length header + bytes, 3 per pixel (same format
    as ComfyUI-TripoSG / ForgeExpress data images)."""
    data = text.encode("utf-8")
    payload = struct.pack(">I", len(data)) + data
    payload += b"\x00" * (-len(payload) % 3)
    side = max(1, math.ceil(math.sqrt(len(payload) // 3)))
    payload += b"\x00" * (side * side * 3 - len(payload))
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(side, side, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


async def _gpu_facts():
    rc, out, _ = await R.run_subprocess(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                                         "--format=csv,noheader"])
    return out.strip() if rc == 0 else f"nvidia-smi rc={rc}"


class RigStudioA2FProbe:
    CATEGORY = "RigStudio"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("report_data", "engine_path")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "hardware_compat": ("BOOLEAN", {"default": True,
                                            "tooltip": "Build an Ampere+ hardware-compatible engine (reusable on all newer GPUs)"}),
            "note": ("STRING", {"default": "", "tooltip": "free text copied into the report"}),
        }}

    async def run(self, hardware_compat, note):
        report = {"note": note, "python": platform.python_version(), "glibc": " ".join(platform.libc_ver()),
                  "steps": []}
        log = lambda m: (report["steps"].append(m), print(f"[RigStudioA2FProbe] {m}", flush=True))
        T = time.time()
        try:
            report["gpu"] = await _gpu_facts()
            rt = await R.ensure_runtime(log)
            report["runtime_timings"] = rt["timings"]
            src, how, secs = await R.ensure_mark_model(log)
            report["model_source"], report["model_s"] = how, secs
            engine = os.path.join(R.WORK_DIR, "engines",
                                  f"mark-v2.3-trt10.13.3-{'ampere_plus' if hardware_compat else 'native'}.trt")
            os.makedirs(os.path.dirname(engine), exist_ok=True)
            report["engine_build_s"] = await R.build_engine(
                rt, os.path.join(src, "network.onnx"), os.path.join(src, "trt_info.json"), engine, hardware_compat, log)
            report["engine_mb"] = round(os.path.getsize(engine) / 1e6, 2)
            model_json = R.model_workdir(src, engine, "mark-v2.3")

            out_dir = os.path.join(R.WORK_DIR, "probe_out")
            os.makedirs(out_dir, exist_ok=True)
            wav = os.path.join(rt["root"], "sample", "audio_4sec_16k_s16le.wav")
            t0 = time.time()
            rc, out, err = await R.run_subprocess([rt["bin"], model_json, out_dir, "--batch", "1", wav],
                                                  stream_prefix="a2f", env=rt["env"])
            report["a2f_s"] = round(time.time() - t0, 2)
            report["a2f_rc"], report["a2f_stdout"], report["a2f_stderr"] = rc, out[-1500:], err[-3000:]
            res = os.path.join(out_dir, "audio_4sec_16k_s16le.a2f.txt")
            if rc == 0 and os.path.isfile(res):
                with open(res) as f:
                    n = int(f.readline())
                    rows = np.array([list(map(float, l.split())) for l in f if l.strip()])
                w = rows[:, 1:]
                report["result"] = {"weights": n, "frames": int(len(rows)),
                                    "ts_step": float(np.median(np.diff(rows[:, 0]))) if len(rows) > 1 else None,
                                    "max_per_weight": [round(float(x), 3) for x in w.max(0)],
                                    "first_frame": [round(float(x), 3) for x in w[0]]}
            report["ok"] = rc == 0
        except Exception as e:  # the report must still come out so the job teaches us something
            import traceback
            report["ok"] = False
            report["error"] = f"{type(e).__name__}: {e}"
            report["traceback"] = traceback.format_exc()[-3000:]
            engine = ""
        report["total_s"] = round(time.time() - T, 2)
        text = json.dumps(report, indent=1)
        print(text, flush=True)
        if not report.get("engine_mb"):
            # never hand an empty path downstream (a failing smuggle node could sink the report)
            engine = os.path.join(R.WORK_DIR, "no-engine.txt")
            os.makedirs(R.WORK_DIR, exist_ok=True)
            with open(engine, "w") as f:
                f.write(report.get("error", "no engine built"))
        return (encode_string_as_image(text), engine)


class RigStudioBuildTrack:
    """Job B: speech audio + intent (emotion keyframes per sentence) -> hybrid expression track.

    A2F-3D (Mark v2.3) driven by the intent's emotion curve owns mouth, jaw and emotional
    expression; analyze.py rule gestures (brow flicks leading pitch accents, blinks, nods,
    head motion, gaze) are added on top. Output: the track JSON as a data image (+ its path)."""
    CATEGORY = "RigStudio"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("track_data", "summary")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_url": ("STRING", {"default": "", "tooltip": "speech audio URL (or local path)"}),
            "intent": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "S <t0> <t1> [q|wq] <emotion> <weight> ... | none — from Job A; empty = neutral"}),
            "engine_url": ("STRING", {"default": "",
                                      "tooltip": "prebuilt Ampere+ TensorRT engine URL; empty = build one (~1-2 min)"}),
            "seed": ("INT", {"default": 7, "min": 0, "max": 2 ** 31 - 1}),
        }}

    async def run(self, audio_url, intent, engine_url, seed):
        T, steps = time.time(), []
        log = lambda m: (steps.append(m), print(f"[RigStudioBuildTrack] {m}", flush=True))
        rt, (aenv, _), (src, _, _) = await asyncio.gather(R.ensure_runtime(log), R.ensure_analysis(log),
                                                          R.ensure_mark_model(log))
        wav, duration = await R.fetch_audio_16k(audio_url.strip(), log)
        job = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rigstudio", "track_job.py")
        wd = os.path.join(R.WORK_DIR, "job")
        os.makedirs(wd, exist_ok=True)
        intent_p = os.path.join(wd, "intent.txt")
        with open(intent_p, "w", encoding="utf-8") as f:
            f.write(intent or "")

        async def analysis(*args):
            rc, out, err = await R.run_subprocess([sys.executable, job, *args], stream_prefix="track", env=aenv)
            if rc != 0:
                raise RuntimeError(f"track_job {args[0]} failed (rc={rc}):\n{err[-3000:]}")
            return out

        # the emotion curve is written next to the wav, where rigstudio-a2f looks for it
        await analysis("emotion", intent_p, f"{duration:.3f}", os.path.splitext(wav)[0] + ".emotion.txt")
        if engine_url.strip():
            engine = await R.fetch_engine(engine_url.strip(), log)
        else:
            engine = os.path.join(R.WORK_DIR, "engines", "mark-v2.3-trt10.13.3-ampere_plus.trt")
            os.makedirs(os.path.dirname(engine), exist_ok=True)
            if not os.path.isfile(engine):
                await R.build_engine(rt, os.path.join(src, "network.onnx"), os.path.join(src, "trt_info.json"),
                                     engine, True, log)
        model_json = R.model_workdir(src, engine, "mark-v2.3")
        a2f_txt = os.path.join(wd, "input16k.a2f.txt")
        if os.path.exists(a2f_txt):                   # warm container: never read a stale result
            os.remove(a2f_txt)
        t0 = time.time()
        rc, out, err = await R.run_subprocess([rt["bin"], model_json, wd, "--batch", "1", wav],
                                              stream_prefix="a2f", env=rt["env"])
        if rc != 0 or not os.path.isfile(a2f_txt):
            raise RuntimeError(f"rigstudio-a2f failed (rc={rc}):\n{out[-1500:]}\n{err[-3000:]}")
        log(f"a2f {time.time() - t0:.1f}s")
        track_p = os.path.join(wd, "track.json")
        await analysis("finish", wav, a2f_txt, intent_p, os.path.join(src, "bs_skin.npz"), track_p,
                       "--seed", str(seed), "--audio-name", os.path.basename(audio_url.strip().split("?")[0]))
        with open(track_p, encoding="utf-8") as f:
            text = f.read()
        summary = json.dumps({"duration_s": round(duration, 2), "track_bytes": len(text),
                              "total_s": round(time.time() - T, 1), "steps": steps})
        print(summary, flush=True)
        return (encode_string_as_image(text), summary)


NODE_CLASS_MAPPINGS = {"RigStudioA2FProbe": RigStudioA2FProbe, "RigStudioBuildTrack": RigStudioBuildTrack}
NODE_DISPLAY_NAME_MAPPINGS = {"RigStudioA2FProbe": "Rig Studio: A2F runtime probe",
                              "RigStudioBuildTrack": "Rig Studio: build expression track"}
