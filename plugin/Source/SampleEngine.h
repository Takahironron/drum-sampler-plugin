#pragma once

#include "KitModel.h"

#include <array>
#include <atomic>

// 音を鳴らすエンジン（計画書のモジュール `Engine` の最小版: SampleVoice、強さに合う音の選択、チョーク）。
// 実時間の決まり: processBlock の中ではメモリ確保・ロック待ち・ファイル入出力をしない。
//
// 鳴らし方は Python の参照実装（analysis/drumeval/extract.py の `render_midi`）と同じで、1 サンプル単位で同じ結果になる
// （analysis/engine_parity.py と CI で確かめる）。鳴らし方を変えるときは両方を変える。
//   1. MIDI の強さ 1〜127 → 目標のピーク dB（その種類の levelLoDb 〜 levelHiDb に dB で線形）。
//   2. その種類の音のうち、ピークが目標に最も近い音を選ぶ（同じ近さなら、ピークの小さい方）。
//   3. 音量を目標に合わせる。持ち上げるのは maxBoostDb まで、下げる側は制限しない。
//   4. 音は頭から鳴らす。立ち上がりは leadSeconds 後に来る（プラグインはこの長さを遅れとしてホストに伝える）。
//   5. チョーク: 同じ種類の次の打の立ち上がり（ノートオンから leadSeconds 後）から、前の音を chokeFadeSeconds で止める（シンバル以外）。
namespace kit
{
    class SampleVoice
    {
    public:
        void start (const Sample* s, double hostRate, float gain, int ownerPart) noexcept;
        // delaySamples 後から、fadeSamples かけて止める。すでに止めている途中なら何もしない
        void choke (int delaySamples, int fadeSamples) noexcept;
        void render (juce::AudioBuffer<float>& out, int startSample, int numSamples) noexcept;
        bool isActive() const noexcept { return sample != nullptr; }
        int owner() const noexcept { return ownerPart; }
        juce::int64 startedAt() const noexcept { return startOrder; }

        static std::atomic<juce::int64> counter;

    private:
        const Sample* sample = nullptr;
        double position = 0.0;
        double increment = 1.0;
        float gain = 1.0f;
        int chokeCountdown = -1; // 0 以上なら、このサンプル数の後にフェードを始める（-1 は止めていない）
        int fadePos = 0;
        int fadeLength = 1;
        int ownerPart = -1;
        juce::int64 startOrder = 0;
    };

    class SampleEngine
    {
    public:
        static constexpr int maxVoices = 32;
        static constexpr double maxBoostDb = 12.0;        // Python の MAX_BOOST_DB
        static constexpr double chokeFadeSeconds = 0.005; // Python の CHOKE_FADE_SEC

        void prepare (double hostRate) noexcept;

        // メッセージスレッドから呼ぶ。音を鳴らすスレッドは次のブロックで新しいキットに切り替える
        void setKit (std::shared_ptr<const KitModel> kit);

        // 音を鳴らすスレッドから。velocity は MIDI の強さ（1〜127）
        void noteOn (int midiNote, int velocity) noexcept;
        void render (juce::AudioBuffer<float>& out, int startSample, int numSamples) noexcept;
        void beginBlock() noexcept; // 保留中のキットを取り込む

        std::shared_ptr<const KitModel> currentKit() const; // メッセージスレッド用（表示）

    private:
        void updateTimings() noexcept;

        std::array<SampleVoice, maxVoices> voices;
        std::shared_ptr<const KitModel> kit;          // 音を鳴らすスレッドが持つ
        std::shared_ptr<const KitModel> pendingKit;   // ロックで守る受け渡し
        juce::SpinLock pendingLock;
        double sampleRate = 44100.0;
        int leadSamples = 0;   // ノートオンから立ち上がりまで（キットの leadSeconds）
        int fadeSamples = 220; // チョークのフェード
    };
}
