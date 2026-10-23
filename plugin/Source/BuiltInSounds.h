#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

// 内蔵の仮の音。音声ファイルをコミットしないため、起動時にコードで合成する。
// 音を鳴らすスレッドでは呼ばない（メモリ確保をする）。
namespace builtin
{
    struct Sound
    {
        juce::String name;
        int midiNote;
        juce::AudioBuffer<float> buffer; // モノラル
    };

    // キック・スネア・ハイハット風の3音（GM ドラムマップの 36 / 38 / 42）
    std::vector<Sound> makeDefaultSounds (double sampleRate);

    // juce::Synthesiser に登録する SamplerSound を作る
    juce::SynthesiserSound::Ptr makeSamplerSound (const Sound& sound, double sampleRate);
}
