// rigstudio-a2f: batch audio -> ARKit blendshape weights with the NVIDIA Audio2Face-3D SDK.
//
//   rigstudio-a2f <model.json> <out_dir> [--diffusion] [--identity N] [--batch N]
//                 [--emotion e0,e1,...,e9] <wav> [<wav> ...]
//   rigstudio-a2f ... --list <file-with-one-wav-path-per-line>
//
// For every 16 kHz mono wav writes <out_dir>/<name>.a2f.txt:
//   line 1: "<weight_count>"
//   then one line per frame: "<timestamp> w0 w1 ... wN-1"   (weights in the model's pose order)
//
// --emotion sets a constant explicit emotion vector (amazement, anger, cheekiness, disgust,
// fear, grief, joy, outofbreath, pain, sadness). Default: neutral (all zero).
// A time-varying emotion track overrides it per file: if "<wav minus .wav>.emotion.txt" exists,
// each line is "<seconds> e0 ... e9" (keyframes, ascending time; the writer should densify to
// the frame rate — A2F samples the accumulated emotion per frame).
// Audio2Emotion is intentionally NOT used here: its output may only feed A2F inside NVIDIA's
// pipeline and must never be read or stored (see docs/expression-track.md).

#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"
#include "AudioFile.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Destroyer { template <typename T> void operator()(T* p) const { p->Destroy(); } };
template <typename T> using UniquePtr = std::unique_ptr<T, Destroyer>;

#define CHECK(expr, what)                                                                 \
  do { if (auto _e = (expr)) { std::cerr << "error: " << what << ": " << _e.message() << "\n"; \
       return false; } } while (0)

std::string Basename(const std::string& p) {
  auto s = p.find_last_of("/\\");
  std::string b = s == std::string::npos ? p : p.substr(s + 1);
  auto d = b.find_last_of('.');
  return d == std::string::npos ? b : b.substr(0, d);
}

// "<seconds> e0 ... eN" per line -> (timestamp in 16 kHz samples, vector)
std::vector<std::pair<long long, std::vector<float>>> ReadEmotionTrack(const std::string& wav) {
  std::vector<std::pair<long long, std::vector<float>>> keys;
  auto dot = wav.find_last_of('.');
  std::ifstream in((dot == std::string::npos ? wav : wav.substr(0, dot)) + ".emotion.txt");
  std::string line;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    double t;
    if (!(ss >> t)) continue;
    std::vector<float> e;
    float v;
    while (ss >> v) e.push_back(v);
    keys.emplace_back(static_cast<long long>(t * 16000.0 + 0.5), std::move(e));
  }
  return keys;
}

bool ReadWav(const std::string& path, std::vector<float>& out) {
  AudioFile<float> f;
  if (!f.load(path)) return false;
  if (f.getSampleRate() != 16000 || f.samples.empty()) {
    std::cerr << "skip (need 16 kHz): " << path << "\n";
    return false;
  }
  out = f.samples[0];
  return true;
}

struct Frame { long long ts; std::vector<float> w; };
struct CallbackData { std::vector<std::vector<Frame>> tracks; };

void OnResults(void* user, const nva2f::IBlendshapeExecutor::HostResults& r, std::error_code err) {
  if (err) { std::cerr << "result error: " << err.message() << "\n"; return; }
  auto& d = *static_cast<CallbackData*>(user);
  Frame f{static_cast<long long>(r.timeStampCurrentFrame),
          std::vector<float>(r.weights.Data(), r.weights.Data() + r.weights.Size())};
  d.tracks[r.trackIndex].push_back(std::move(f));
}

bool RunBatch(const std::vector<std::string>& wavs, const std::string& model, const std::string& outDir,
              bool diffusion, std::size_t identity, const std::vector<float>& emotion) {
  std::vector<std::vector<float>> audio(wavs.size());
  for (std::size_t i = 0; i < wavs.size(); ++i)
    if (!ReadWav(wavs[i], audio[i])) audio[i].assign(1600, 0.0f);   // keep track indices aligned

  const auto opt = nva2f::IGeometryExecutor::ExecutionOption::Skin;
  UniquePtr<nva2f::IBlendshapeExecutorBundle> bundle(
      diffusion ? nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(wavs.size(), model.c_str(), opt, false,
                                                                    identity, false, nullptr, nullptr)
                : nva2f::ReadRegressionBlendshapeSolveExecutorBundle(wavs.size(), model.c_str(), opt, false,
                                                                     30, 1, nullptr, nullptr));
  if (!bundle) { std::cerr << "error: could not load " << model << "\n"; return false; }
  auto& ex = bundle->GetExecutor();
  if (ex.GetResultType() != nva2f::IBlendshapeExecutor::ResultsType::HOST) {
    std::cerr << "error: expected host results (CPU solver)\n"; return false;
  }
  CallbackData data;
  data.tracks.resize(wavs.size());
  CHECK(ex.SetResultsCallback(&OnResults, &data), "set callback");

  auto stream = bundle->GetCudaStream().Data();
  for (std::size_t t = 0; t < wavs.size(); ++t) {
    CHECK(bundle->GetAudioAccumulator(t).Accumulate(
              nva2x::HostTensorFloatConstView{audio[t].data(), audio[t].size()}, stream), "audio");
    CHECK(bundle->GetAudioAccumulator(t).Close(), "close audio");
    auto& emo = bundle->GetEmotionAccumulator(t);
    auto keys = ReadEmotionTrack(wavs[t]);
    if (keys.empty()) keys.emplace_back(0, emotion);
    for (const auto& [ts, vals] : keys) {
      std::vector<float> e(emo.GetEmotionSize(), 0.0f);
      for (std::size_t k = 0; k < e.size() && k < vals.size(); ++k) e[k] = vals[k];
      CHECK(emo.Accumulate(ts, nva2x::HostTensorFloatConstView{e.data(), e.size()}, stream), "emotion");
    }
    CHECK(emo.Close(), "close emotion");
  }
  while (nva2x::GetNbReadyTracks(ex) > 0) CHECK(ex.Execute(nullptr), "execute");
  for (std::size_t t = 0; t < wavs.size(); ++t) CHECK(ex.Wait(t), "wait");

  const std::size_t n = ex.GetWeightCount();
  for (std::size_t t = 0; t < wavs.size(); ++t) {
    std::ofstream o(outDir + "/" + Basename(wavs[t]) + ".a2f.txt");
    o << n << "\n";
    for (const auto& f : data.tracks[t]) {
      o << f.ts;
      for (float w : f.w) o << ' ' << w;
      o << '\n';
    }
  }
  std::cout << "batch of " << wavs.size() << " done (" << n << " weights)\n" << std::flush;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: rigstudio-a2f <model.json> <out_dir> [--diffusion] [--identity N] [--batch N]"
                 " [--emotion e0,...,e9] [--list file] <wav>...\n";
    return 2;
  }
  std::string model = argv[1], outDir = argv[2];
  bool diffusion = false;
  std::size_t identity = 0, batch = 8;
  std::vector<float> emotion;
  std::vector<std::string> wavs;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--diffusion") diffusion = true;
    else if (a == "--identity" && i + 1 < argc) identity = std::stoul(argv[++i]);
    else if (a == "--batch" && i + 1 < argc) batch = std::stoul(argv[++i]);
    else if (a == "--emotion" && i + 1 < argc) {
      std::stringstream ss(argv[++i]); std::string tok;
      while (std::getline(ss, tok, ',')) emotion.push_back(std::stof(tok));
    } else if (a == "--list" && i + 1 < argc) {
      std::ifstream in(argv[++i]); std::string line;
      while (std::getline(in, line)) if (!line.empty()) wavs.push_back(line);
    } else wavs.push_back(a);
  }
  std::size_t failed = 0;
  for (std::size_t i = 0; i < wavs.size(); i += batch) {
    std::vector<std::string> chunk(wavs.begin() + i, wavs.begin() + std::min(wavs.size(), i + batch));
    if (!RunBatch(chunk, model, outDir, diffusion, identity, emotion)) ++failed;
  }
  std::cout << "done: " << wavs.size() << " files, " << failed << " failed batches\n";
  return failed ? 1 : 0;
}
