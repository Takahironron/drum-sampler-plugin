#include "Transcriber.h"

#include <onnxruntime_cxx_api.h>

#include <cstdlib>
#include <stdexcept>

namespace drum::analysis
{
    namespace
    {
        /** ONNX Runtime の公式の配布物（1.31.0 で確認）は、利用状況を Microsoft に送る機能（テレメトリ）が既定で有効
            （配布物の Privacy.md）。初期化の前に環境変数 ORT_DISABLE_TELEMETRY=1 を置くと、そのプロセスでは送信の仕組みも
            端末の識別子も作られない。利用者の同意なしに外へ送らないよう、必ず切る。 */
        bool disableTelemetryBeforeInit()
        {
           #if defined (_WIN32)
            _putenv_s ("ORT_DISABLE_TELEMETRY", "1");
           #else
            setenv ("ORT_DISABLE_TELEMETRY", "1", 1);
           #endif
            return true;
        }
    }

    struct Transcriber::Impl
    {
        bool telemetryOff = disableTelemetryBeforeInit(); // env より前に初期化する（メンバーの並びの順）
        Ort::Env env { ORT_LOGGING_LEVEL_WARNING, "drum-transcriber" };
        Ort::SessionOptions options;
        std::unique_ptr<Ort::Session> session;
        std::string inputName, outputName;
    };

    Transcriber::Transcriber (const void* modelBytes, size_t modelSize, ModelInfo info, int numThreads)
        : impl_ (std::make_unique<Impl>()), info_ (std::move (info))
    {
        try
        {
            impl_->env.DisableTelemetryEvents(); // 環境変数に加えて、API でも切る
            if (numThreads > 0)
                impl_->options.SetIntraOpNumThreads (numThreads);
            impl_->options.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
            impl_->session = std::make_unique<Ort::Session> (impl_->env, modelBytes, modelSize, impl_->options);
            Ort::AllocatorWithDefaultOptions allocator;
            impl_->inputName = impl_->session->GetInputNameAllocated (0, allocator).get();
            impl_->outputName = impl_->session->GetOutputNameAllocated (0, allocator).get();
        }
        catch (const Ort::Exception& e)
        {
            throw std::runtime_error (std::string ("ONNX Runtime: ") + e.what());
        }
    }

    Transcriber::~Transcriber() = default;

    std::vector<float> Transcriber::run (const LogMel& features, int& numFramesOut) const
    {
        numFramesOut = 0;
        if (features.numFrames <= 0)
            return {};
        try
        {
            const auto memory = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
            const int64_t shape[3] = { 1, features.numFrames, features.numMels };
            // ONNX Runtime は入力を書き換えないが、API が const を取らない
            auto input = Ort::Value::CreateTensor<float> (memory, const_cast<float*> (features.data.data()), features.data.size(), shape, 3);
            const char* inputNames[] = { impl_->inputName.c_str() };
            const char* outputNames[] = { impl_->outputName.c_str() };
            auto outputs = impl_->session->Run (Ort::RunOptions { nullptr }, inputNames, &input, 1, outputNames, 1);
            const auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, frames, classes]
            if (outShape.size() != 3 || outShape[0] != 1 || outShape[2] != static_cast<int64_t> (info_.classes.size()))
                throw std::runtime_error ("ONNX Runtime: モデルの出力の形が [1, frames, classes] ではない");
            numFramesOut = static_cast<int> (outShape[1]);
            const float* data = outputs[0].GetTensorData<float>();
            return std::vector<float> (data, data + static_cast<size_t> (outShape[1] * outShape[2]));
        }
        catch (const Ort::Exception& e)
        {
            throw std::runtime_error (std::string ("ONNX Runtime: ") + e.what());
        }
    }

    Transcription Transcriber::transcribe (const std::vector<double>& mono, int sampleRate) const
    {
        Transcription out;
        const auto x = resampleTo (mono, sampleRate, info_.features.sampleRate);
        const auto features = logMel (x, info_.features);
        out.probabilities = run (features, out.numFrames);
        out.hits = pickHits (out.probabilities.data(), out.numFrames, static_cast<int> (info_.classes.size()), info_.features, info_.thresholds);
        return out;
    }
}
