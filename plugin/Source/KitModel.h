#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

#include <memory>
#include <vector>

// キットのデータ（計画書のモジュール `KitModel`）。
// Python の analysis/drumeval/extract.py が書く kit.json（種類ごとの代表音の WAV、ピーク dB、強さと音量の対応）を読む。
// 読み込みはメッセージスレッドで行い、音を鳴らすスレッドでは読むだけ（メモリ確保・ファイル入出力をしない）。
//
// キットと MIDI の約束（Python の `drumeval.extract` と同じ。変えるときは両方を変える）:
//   - MIDI の強さ 1〜127 は、種類ごとの「打のピークの範囲」(levelLoDb 〜 levelHiDb) に dB で線形に対応する（kit.json の level_range_db）。
//   - 各音は、打点の leadSeconds だけ前から始まる（立ち上がりは音の頭からこの位置。kit.json の lead_sec）。
namespace kit
{
    struct Sample
    {
        juce::AudioBuffer<float> buffer; // モノラル
        double sampleRate = 44100.0;
        double peakDb = 0.0;
        juce::String file;
    };

    struct Part
    {
        juce::String cls;                 // kick / snare / hihat / tom / cymbal
        int midiNote = 36;                // GM ドラムマップの代表ノート
        bool choke = false;               // 同じ種類の次の打で前の音を止める（シンバル以外）
        std::vector<Sample> samples;      // ピークの小さい順（同じピークなら kit.json の順）
        double minPeakDb = 0.0, maxPeakDb = 0.0;   // 音のピークの範囲（表示用）
        double levelLoDb = -30.0, levelHiDb = 0.0; // 強さ 1 と 127 に対応する打のピーク
    };

    class KitModel
    {
    public:
        static constexpr double velocityRangeDb = 30.0;  // 範囲の指定が無いとき: 最も大きい音から、この幅までを強さで表す
        static constexpr double defaultLeadSeconds = 0.002;

        // kit.json を読み、WAV をメモリに載せる。失敗したら false と理由。
        bool loadFromJson (const juce::File& kitJson, juce::String& error);

        // コードで合成した仮のキット（kit.json が無いときの既定）
        static std::shared_ptr<const KitModel> makeBuiltIn (double sampleRate);

        const Part* partForNote (int midiNote) const noexcept;
        const std::vector<Part>& parts() const noexcept { return parts_; }
        juce::String name() const { return name_; }
        double leadSeconds() const noexcept { return leadSec_; }

        // MIDI の強さ（1〜127）→ 目標のピーク dB。Python の `level_for_velocity` と同じ式
        static double levelForVelocity (int velocity, double loDb, double hiDb) noexcept;

        // GM ドラムマップのノート → 種類
        static const char* classForNote (int midiNote) noexcept;
        static int noteForClass (const juce::String& cls) noexcept;

    private:
        std::vector<Part> parts_;
        juce::String name_;
        double leadSec_ = 0.0;
    };
}
