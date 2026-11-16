// kit.json を読み、MIDI の打点を鳴らして WAV に書く確認用のプログラム（DAW 無しでエンジンを通す）。
//
// 使い方:
//   KitRenderTest <kit.json> <out.wav> [秒数]
//       決まった 8 ビートを鳴らす。ピークを 0.9 にそろえた 16 bit ステレオ。
//   KitRenderTest <kit.json> <out.wav> --events <events.csv> [秒数]
//       打点のリスト（1 行に「サンプル位置,ノート番号,強さ」。# で始まる行は読み飛ばす）を鳴らす。
//       キットのサンプリング周波数のまま、音量をそろえずに 32 bit float のモノラルで書く。
//       Python の参照実装（analysis/drumeval/extract.py の render_midi）と 1 サンプル単位で比べるためのもの（analysis/engine_parity.py）。
#include <juce_audio_utils/juce_audio_utils.h>

#include "../Source/KitModel.h"
#include "../Source/SampleEngine.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace
{
    struct Event
    {
        int sample = 0;
        int note = 0;
        int velocity = 0;
    };

    bool readEvents (const juce::File& file, std::vector<Event>& events)
    {
        if (! file.existsAsFile())
            return false;
        juce::StringArray lines;
        file.readLines (lines);
        for (const auto& raw : lines)
        {
            const auto line = raw.trim();
            if (line.isEmpty() || line.startsWithChar ('#'))
                continue;
            juce::StringArray cols;
            cols.addTokens (line, ",", "");
            if (cols.size() < 3 || ! cols[0].trim().containsOnly ("0123456789"))
                continue; // 見出しの行
            events.push_back ({ cols[0].trim().getIntValue(), cols[1].trim().getIntValue(), cols[2].trim().getIntValue() });
        }
        std::stable_sort (events.begin(), events.end(), [] (const Event& a, const Event& b) { return a.sample < b.sample; });
        return true;
    }

    void renderEvents (kit::SampleEngine& engine, const std::vector<Event>& events, juce::AudioBuffer<float>& out)
    {
        const int block = 512;
        const int total = out.getNumSamples();
        size_t next = 0;
        for (int pos = 0; pos < total; pos += block)
        {
            const int n = juce::jmin (block, total - pos);
            engine.beginBlock();
            int cursor = pos;
            while (next < events.size() && events[next].sample < pos + n)
            {
                const auto& e = events[next++];
                if (e.sample > cursor)
                {
                    engine.render (out, cursor, e.sample - cursor);
                    cursor = e.sample;
                }
                engine.noteOn (e.note, e.velocity);
            }
            if (pos + n > cursor)
                engine.render (out, cursor, pos + n - cursor);
        }
    }

    bool writeWav (const juce::File& file, const juce::AudioBuffer<float>& buffer, double sampleRate, int bits)
    {
        juce::WavAudioFormat wav;
        file.deleteFile();
        std::unique_ptr<juce::OutputStream> stream = std::make_unique<juce::FileOutputStream> (file);
        auto writer = wav.createWriterFor (stream, juce::AudioFormatWriterOptions().withSampleRate (sampleRate)
                                                       .withNumChannels (buffer.getNumChannels()).withBitsPerSample (bits));
        if (writer == nullptr)
            return false;
        return writer->writeFromAudioSampleBuffer (buffer, 0, buffer.getNumSamples());
    }
}

int main (int argc, char** argv)
{
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String (juce::CharPointer_UTF8 (argv[i])));
    juce::String eventsPath;
    const int flag = args.indexOf ("--events");
    if (flag >= 0)
    {
        if (flag + 1 >= args.size())
        {
            std::cerr << "--events の後にファイル名が要る\n";
            return 2;
        }
        eventsPath = args[flag + 1];
        args.removeRange (flag, 2);
    }
    if (args.size() < 2)
    {
        std::cerr << "使い方: KitRenderTest <kit.json> <out.wav> [--events <events.csv>] [秒数]\n";
        return 2;
    }
    const auto cwd = juce::File::getCurrentWorkingDirectory();
    const juce::File kitJson = cwd.getChildFile (args[0]);
    const juce::File outFile = cwd.getChildFile (args[1]);
    double seconds = args.size() > 2 ? args[2].getDoubleValue() : 0.0;

    auto model = std::make_shared<kit::KitModel>();
    juce::String error;
    if (! model->loadFromJson (kitJson, error))
    {
        std::cerr << "読み込み失敗: " << error << "\n";
        return 1;
    }
    double longest = 0.0;
    for (const auto& p : model->parts())
    {
        std::cout << p.cls << ": " << p.samples.size() << " 音 (" << p.minPeakDb << " .. " << p.maxPeakDb << " dB)  強さ 1..127 = "
                  << p.levelLoDb << " .. " << p.levelHiDb << " dB" << (p.choke ? "  choke" : "") << "\n";
        for (const auto& s : p.samples)
            longest = juce::jmax (longest, s.buffer.getNumSamples() / s.sampleRate);
    }

    const bool exact = eventsPath.isNotEmpty();
    const double sr = exact ? model->parts().front().samples.front().sampleRate : 44100.0;
    std::vector<Event> events;
    if (exact)
    {
        if (! readEvents (cwd.getChildFile (eventsPath), events) || events.empty())
        {
            std::cerr << "打点のリストを読めない: " << eventsPath << "\n";
            return 1;
        }
        if (seconds <= 0.0)
            seconds = events.back().sample / sr + longest + 0.5;
    }
    else
    {
        if (seconds <= 0.0)
            seconds = 8.0;
        // 120 BPM の 8 ビート: キック 1・3 拍、スネア 2・4 拍、ハイハット 8 分、小節の頭にシンバル
        const double beat = 0.5;
        for (double t = 0.0; t < seconds; t += beat * 4)
        {
            auto add = [&] (double at, int note, int vel) { events.push_back ({ (int) (at * sr), note, vel }); };
            add (t, 36, 110); add (t + 2 * beat, 36, 95); add (t + 2.5 * beat, 36, 80);
            add (t + beat, 38, 115); add (t + 3 * beat, 38, 100);
            for (int k = 0; k < 8; ++k) add (t + k * beat / 2, 42, k % 2 == 0 ? 90 : 60);
            add (t, 49, 110);
        }
        std::stable_sort (events.begin(), events.end(), [] (const Event& a, const Event& b) { return a.sample < b.sample; });
    }

    kit::SampleEngine engine;
    engine.prepare (sr);
    engine.setKit (model);

    const int total = (int) (seconds * sr);
    juce::AudioBuffer<float> out (exact ? 1 : 2, total);
    out.clear();
    renderEvents (engine, events, out);

    const float peak = out.getMagnitude (0, total);
    if (! exact && peak > 0.0f)
        out.applyGain (0.9f / peak);
    if (! writeWav (outFile, out, sr, exact ? 32 : 16))
    {
        std::cerr << "WAV を書けない\n";
        return 1;
    }
    std::cout << "書き出した: " << outFile.getFullPathName() << "  打 " << events.size() << "  ピーク " << peak
              << "  立ち上がりまで " << model->leadSeconds() * 1000.0 << " ms\n";
    return peak > 0.0f ? 0 : 1;
}
