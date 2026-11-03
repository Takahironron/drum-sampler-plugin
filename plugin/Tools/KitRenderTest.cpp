// kit.json を読み、決まった MIDI パターンを鳴らして WAV に書く確認用のプログラム（DAW 無しでエンジンを通す）。
// 使い方: KitRenderTest <kit.json> <out.wav> [秒数]
#include <juce_audio_utils/juce_audio_utils.h>

#include "../Source/KitModel.h"
#include "../Source/SampleEngine.h"

int main (int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "使い方: KitRenderTest <kit.json> <out.wav> [秒数]\n";
        return 2;
    }
    const juce::File kitJson { juce::String (argv[1]) };
    const juce::File outFile { juce::String (argv[2]) };
    const double seconds = argc > 3 ? juce::String (argv[3]).getDoubleValue() : 8.0;
    const double sr = 44100.0;

    auto model = std::make_shared<kit::KitModel>();
    juce::String error;
    if (! model->loadFromJson (kitJson, error))
    {
        std::cerr << "読み込み失敗: " << error << "\n";
        return 1;
    }
    for (const auto& p : model->parts())
        std::cout << p.cls << ": " << p.samples.size() << " 音 (" << p.minPeakDb << " .. " << p.maxPeakDb << " dB)" << (p.choke ? " choke" : "") << "\n";

    kit::SampleEngine engine;
    engine.prepare (sr);
    engine.setKit (model);

    // 120 BPM の 8 ビート: キック 1・3 拍、スネア 2・4 拍、ハイハット 8 分、小節の頭にシンバル
    const double beat = 0.5;
    juce::MidiBuffer midi;
    for (double t = 0.0; t < seconds; t += beat * 4)
    {
        auto add = [&] (double at, int note, int vel) { midi.addEvent (juce::MidiMessage::noteOn (10, note, (juce::uint8) vel), (int) (at * sr)); };
        add (t, 36, 110); add (t + 2 * beat, 36, 95); add (t + 2.5 * beat, 36, 80);
        add (t + beat, 38, 115); add (t + 3 * beat, 38, 100);
        for (int k = 0; k < 8; ++k) add (t + k * beat / 2, 42, k % 2 == 0 ? 90 : 60);
        add (t, 49, 110);
    }

    const int block = 512;
    const int total = (int) (seconds * sr);
    juce::AudioBuffer<float> out (2, total);
    out.clear();
    for (int pos = 0; pos < total; pos += block)
    {
        const int n = juce::jmin (block, total - pos);
        engine.beginBlock();
        int cursor = pos;
        for (const auto meta : midi)
        {
            if (meta.samplePosition < pos || meta.samplePosition >= pos + n)
                continue;
            if (meta.samplePosition > cursor)
            {
                engine.render (out, cursor, meta.samplePosition - cursor);
                cursor = meta.samplePosition;
            }
            engine.noteOn (meta.getMessage().getNoteNumber(), meta.getMessage().getFloatVelocity());
        }
        if (pos + n > cursor)
            engine.render (out, cursor, pos + n - cursor);
    }
    const float peak = out.getMagnitude (0, total);
    if (peak > 0.0f)
        out.applyGain (0.9f / peak);

    juce::WavAudioFormat wav;
    outFile.deleteFile();
    std::unique_ptr<juce::AudioFormatWriter> writer (wav.createWriterFor (new juce::FileOutputStream (outFile), sr, 2, 16, {}, 0));
    if (writer == nullptr)
    {
        std::cerr << "WAV を書けない\n";
        return 1;
    }
    writer->writeFromAudioSampleBuffer (out, 0, total);
    writer.reset();
    std::cout << "書き出した: " << outFile.getFullPathName() << "  ピーク " << peak << "\n";
    return peak > 0.0f ? 0 : 1;
}
