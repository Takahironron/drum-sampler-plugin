// 採譜モデルの入力（対数メル）と、出力の確率から打点を拾う処理。Python の参照実装と同じ結果を出す。
//
//   間引き        analysis/drumeval/infer.py の resample_to（scipy.signal.resample_poly、窓は kaiser 5.0）
//   対数メル      analysis/drumeval/features.py の log_mel（scipy.signal.stft: 周期的な Hann 窓、両端に n_fft/2 の 0、
//                 1/窓の和 の倍率 → パワー → 三角のメルフィルタ → log(1 + 1000·x)）
//   打点を拾う    analysis/drumeval/features.py の pick_hits（しきい値 ＋ 前後 gap フレームの最大 ＋ 最小間隔）
//
// JUCE に依らない（標準ライブラリだけ）。解析は裏のスレッドで動かす。オーディオスレッドからは呼ばない。
// Python と突き合わせる検査は analysis/transcribe_parity.py（CI: transcribe-parity）。変えるときは両方を変える。
#pragma once

#include <string>
#include <vector>

namespace drum::analysis
{
    /** 対数メルの設定。モデルの json の "features" と同じ（Python の `FeatureConfig`）。 */
    struct FeatureConfig
    {
        int sampleRate = 22050;
        int fftSize = 1024;
        int hop = 220; // 約 10 ms
        int numMels = 64;
        double fMin = 30.0;
        double fMax = 11000.0;

        double frameRate() const noexcept { return static_cast<double> (sampleRate) / hop; }
    };

    /** 行優先の [numFrames × numMels]（float32）。 */
    struct LogMel
    {
        int numFrames = 0;
        int numMels = 0;
        std::vector<float> data;
    };

    /** 拾った打点。`classIndex` はモデルの出力の列（json の "classes" の順）。 */
    struct Onset
    {
        double time = 0.0; // 秒
        int classIndex = 0;
        float probability = 0.0f;
    };

    /** `scipy.signal.resample_poly (x, up, down)` と同じ（多相の FIR。up と down は約分して使う）。 */
    std::vector<double> resamplePoly (const std::vector<double>& x, int up, int down);

    /** `sampleRate` の音を `targetRate` に直す（同じなら そのまま返す）。 */
    std::vector<double> resampleTo (const std::vector<double>& x, int sampleRate, int targetRate);

    /** [numMels × (fftSize/2 + 1)] の三角のメルフィルタ（行優先）。 */
    std::vector<double> melFilterbank (const FeatureConfig& config);

    /** 対数メル。入力は `config.sampleRate` のモノラル。 */
    LogMel logMel (const std::vector<double>& x, const FeatureConfig& config);

    /** 確率（行優先の [numFrames × numClasses]）から打点を拾う。`thresholds` は種類ごと（足りない分は 0.5）。時刻の順に返す。 */
    std::vector<Onset> pickHits (const float* probabilities, int numFrames, int numClasses, const FeatureConfig& config,
                                 const std::vector<float>& thresholds, double minGapSeconds = 0.05);
}
