"""Standalone ONNX -> TensorRT engine builder (run with the private TensorRT on PYTHONPATH).

    python trt_build.py network.onnx trt_info.json out.trt <hardware_compat 0|1>

Mirrors NVIDIA's audio2x/trt.py (trtexec) using the Python API: dynamic-shape profile from
trt_info.json ("--minShapes=input:1x1x8320,emotion:1x1x26", {MAX_BATCH_SIZE} placeholders
filled from its defaults), optional --hardwareCompatibilityLevel=ampere+ so one engine runs
on every Ampere-or-newer GPU.
"""
import json
import re
import sys
import time

try:
    import tensorrt as trt
except ImportError:
    import tensorrt_bindings as trt


def shapes(info):
    d = info.get("defaults", {})
    out = {}
    for arg in info["trt_build_param"]["batch"]:
        for k, v in d.items():
            arg = arg.replace("{" + k + "}", str(v))
        m = re.match(r"--(min|opt|max)Shapes=(.*)", arg)
        for part in m.group(2).split(","):
            name, dims = part.split(":")
            out.setdefault(name, {})[m.group(1)] = tuple(int(x) for x in dims.split("x"))
    return out


def main():
    onnx, info_p, out_p, hw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
    t0 = time.time()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx):
        for i in range(parser.num_errors):
            print("onnx parse error:", parser.get_error(i))
        sys.exit(2)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    for name, s in shapes(json.load(open(info_p))).items():
        profile.set_shape(name, s["min"], s["opt"], s["max"])
        print(f"profile {name}: {s['min']} / {s['opt']} / {s['max']}")
    config.add_optimization_profile(profile)
    if hw:
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    print(f"tensorrt {trt.__version__}, hardware_compat={hw}, parsed in {time.time() - t0:.1f}s", flush=True)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        print("build_serialized_network returned None")
        sys.exit(3)
    with open(out_p, "wb") as f:
        f.write(blob)
    print(f"engine written: {out_p} ({blob.nbytes / 1e6:.1f} MB) total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
