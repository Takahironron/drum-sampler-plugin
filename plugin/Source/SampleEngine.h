#pragma once

#include "KitModel.h"

#include <array>
#include <atomic>

// 音を鳴らすエンジン（計画書のモジュール `Engine` の最小版: SampleVoice、層の選択、ラウンドロビン、チョーク）。
// 実時間の決まり: processBlock の中ではメモリ確保・ロック待ち・ファイル入出力をしない。
namespace kit
{
    class SampleVoice
    {
    public:
        void start (const Sample* s, double hostRate, float gain, int ownerPart) noexcept;
        void stop (double hostRate) noexcept;   // 5 ms で止める（チョーク）
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
        float fade = 1.0f;       // チョーク中は 1 → 0
        float fadeStep = 0.0f;
        int ownerPart = -1;
        juce::int64 startOrder = 0;
    };

    class SampleEngine
    {
    public:
        static constexpr int maxVoices = 32;

        void prepare (double hostRate) noexcept { sampleRate = hostRate; }

        // メッセージスレッドから呼ぶ。音を鳴らすスレッドは次のブロックで新しいキットに切り替える
        void setKit (std::shared_ptr<const KitModel> kit);

        // 音を鳴らすスレッドから
        void noteOn (int midiNote, float velocity01) noexcept;
        void render (juce::AudioBuffer<float>& out, int startSample, int numSamples) noexcept;
        void beginBlock() noexcept; // 保留中のキットを取り込む

        std::shared_ptr<const KitModel> currentKit() const; // メッセージスレッド用（表示）

    private:
        std::array<SampleVoice, maxVoices> voices;
        std::array<int, 16> roundRobin {};
        std::shared_ptr<const KitModel> kit;          // 音を鳴らすスレッドが持つ
        std::shared_ptr<const KitModel> pendingKit;   // ロックで守る受け渡し
        juce::SpinLock pendingLock;
        double sampleRate = 44100.0;
    };
}
