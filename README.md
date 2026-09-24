# ComfyUI-RigStudio

ComfyUI nodes for Rig Studio: build-time facial animation for long-form talking-head content.

## Nodes

| Node | What it does |
|---|---|
| `RigStudioA2FProbe` | Provisions an isolated NVIDIA Audio2Face-3D runtime, builds a TensorRT engine for Audio2Face-3D v2.3 Mark (optionally Ampere+ hardware-compatible), runs it on a 4 s sample and returns a JSON report as a data image. |

## Runtime (no compilation on the host)

* `rigstudio-a2f` + `libaudio2x.so` come from this repo's release `a2f-runtime-v1`
  (built from the MIT-licensed [NVIDIA Audio2Face-3D-SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK);
  third-party licences are inside the tarball).
* TensorRT 10.13.3.9 and CUDA 12 runtime / cuBLAS / cuRAND are installed with `pip --target` into a
  private folder at first use; ComfyUI's own environment is not modified.
* Model: [nvidia/Audio2Face-3D-v2.3-Mark](https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark)
  (NVIDIA Open Model License), staged under `models/audio2face/mark-v2.3/` or downloaded on demand.
* Audio2Emotion is deliberately not used: its licence restricts it to NVIDIA's own A2E→A2F pipeline and
  forbids emotion recognition. Emotion comes from explicit, authored emotion keyframes instead.

Requires Linux x86_64, glibc ≥ 2.35, an NVIDIA driver supporting CUDA 12.9.
