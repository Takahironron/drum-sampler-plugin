#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

#include <memory>
#include <vector>

// キットのデータ（計画書のモジュール `KitModel`）。
// Python の analysis/drumeval/extract.py が書く kit.json（種類ごとの代表音の WAV、ピーク dB、特徴）を読む。
// 読み込みはメッセージスレッドで行い、音を鳴らすスレッドでは読むだけ（メモリ確保・ファイル入出力をしない）。
namespace kit
{
    struct Sample
    {
        juce::AudioBuffer<float> buffer; // モノラル
        double sampleRate = 44100.0;
        float peakDb = 0.0f;
        juce::String file;
    };

    struct Part
    {
        juce::String cls;                 // kick / snare / hihat / tom / cymbal
        int midiNote = 36;                // GM ドラムマップの代表ノート
        bool choke = false;               // 同じ種類の次の打で前の音を止める（ハイハット）
        std::vector<Sample> samples;      // ピークの小さい順
        float minPeakDb = 0.0f, maxPeakDb = 0.0f;
    };

    class KitModel
    {
    public:
        // kit.json を読み、WAV をメモリに載せる。失敗したら false と理由。
        bool loadFromJson (const juce::File& kitJson, juce::String& error);

        // コードで合成した仮のキット（kit.json が無いときの既定）
        static std::shared_ptr<const KitModel> makeBuiltIn (double sampleRate);

        const Part* partForNote (int midiNote) const noexcept;
        const std::vector<Part>& parts() const noexcept { return parts_; }
        juce::String name() const { return name_; }

        // GM ドラムマップのノート → 種類
        static const char* classForNote (int midiNote) noexcept;
        static int noteForClass (const juce::String& cls) noexcept;

    private:
        std::vector<Part> parts_;
        juce::String name_;
    };
}
