// 採譜: ドラムのステム（モノラル）→ 打点（時刻・種類）。ONNX のモデルを ONNX Runtime（MIT）で動かす。
// Python の `drumeval.infer.transcribe_onnx` と同じ結果を出す（analysis/transcribe_parity.py で突き合わせる）。
//
// 推論は重いので、裏のスレッドで呼ぶ。オーディオスレッドからは呼ばない。
#pragma once

#include "Features.h"

#include <memory>
#include <string>
#include <vector>

namespace drum::analysis
{
    /** モデルの json（.onnx と同じ名前）の中身。 */
    struct ModelInfo
    {
        FeatureConfig features;
        std::vector<std::string> classes; // 出力の列の順（例: kick, snare, hihat, tom, cymbal）
        std::vector<float> thresholds;    // 種類ごとのしきい値（classes と同じ順）
    };

    struct Transcription
    {
        std::vector<Onset> hits;
        int numFrames = 0;
        std::vector<float> probabilities; // 行優先の [numFrames × classes]
    };

    class Transcriber
    {
    public:
        /** `modelBytes` は .onnx の中身。読めなければ std::runtime_error を投げる。 */
        Transcriber (const void* modelBytes, size_t modelSize, ModelInfo info, int numThreads = 0);
        ~Transcriber();

        Transcriber (const Transcriber&) = delete;
        Transcriber& operator= (const Transcriber&) = delete;

        const ModelInfo& info() const noexcept { return info_; }

        /** 対数メル（[frames × mels]）→ 確率（[frames × classes]）。 */
        std::vector<float> run (const LogMel& features, int& numFramesOut) const;

        /** `sampleRate` のモノラルの音 → 打点。間引き、対数メル、推論、打点を拾う、を順に行う。 */
        Transcription transcribe (const std::vector<double>& mono, int sampleRate) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        ModelInfo info_;
    };
}
