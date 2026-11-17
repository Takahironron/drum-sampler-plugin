// ドラムのステム（WAV）を採譜して、途中の値と打点を書き出す確認用のプログラム（DAW 無しで解析の入口を通す）。
//
// 使い方:
//   TranscribeTest <model.onnx> <in.wav> <出力フォルダ> [--threads N]
//
// 出力フォルダにできるもの:
//   features.f32   対数メル（float32、行優先の [フレーム × メル]）
//   probs.f32      モデルの出力の確率（float32、行優先の [フレーム × 種類]）
//   hits.csv       打点（time_sec,class,probability）
//   info.json      フレーム数、メルの数、種類、音の長さ、各段の所要時間（ミリ秒）
//
// モデルの設定（対数メルの設定、種類、しきい値）は、.onnx と同じ名前の .json から読む（Python の drumeval.infer.load_meta と同じ）。
// Python の参照実装（analysis/drumeval/infer.py の transcribe_onnx）と突き合わせるためのもの（analysis/transcribe_parity.py）。
#include <juce_audio_utils/juce_audio_utils.h>

#include "../Source/Analysis/Transcriber.h"

#include <iostream>

namespace
{
    using namespace drum::analysis;

    bool readModelInfo (const juce::File& jsonFile, ModelInfo& info, juce::String& error)
    {
        const auto root = juce::JSON::parse (jsonFile);
        if (! root.isObject())
        {
            error = "モデルの json を読めない: " + jsonFile.getFullPathName();
            return false;
        }
        const auto features = root["features"];
        if (features.isObject())
        {
            info.features.sampleRate = static_cast<int> (features.getProperty ("sr", info.features.sampleRate));
            info.features.fftSize = static_cast<int> (features.getProperty ("n_fft", info.features.fftSize));
            info.features.hop = static_cast<int> (features.getProperty ("hop", info.features.hop));
            info.features.numMels = static_cast<int> (features.getProperty ("n_mels", info.features.numMels));
            info.features.fMin = static_cast<double> (features.getProperty ("f_min", info.features.fMin));
            info.features.fMax = static_cast<double> (features.getProperty ("f_max", info.features.fMax));
        }
        if (const auto* classes = root["classes"].getArray())
            for (const auto& c : *classes)
                info.classes.push_back (c.toString().toStdString());
        if (info.classes.empty())
        {
            error = "モデルの json に classes が無い";
            return false;
        }
        // しきい値: "thresholds"（種類別）があればそれ、無ければ "threshold"（1 つ）、どちらも無ければ 0.5
        const float common = static_cast<float> (static_cast<double> (root.getProperty ("threshold", 0.5)));
        const auto perClass = root["thresholds"];
        for (const auto& c : info.classes)
        {
            float t = common;
            if (perClass.isObject() && perClass.hasProperty (juce::Identifier (c)))
                t = static_cast<float> (static_cast<double> (perClass[juce::Identifier (c)]));
            else if (perClass.isObject())
                t = 0.5f; // Python の pick_hits と同じ: 辞書に無い種類は 0.5
            info.thresholds.push_back (t);
        }
        return true;
    }

    bool readMono (const juce::File& wav, std::vector<double>& mono, int& sampleRate, juce::String& error)
    {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (wav));
        if (reader == nullptr)
        {
            error = "音声ファイルを読めない: " + wav.getFullPathName();
            return false;
        }
        const int numChannels = static_cast<int> (reader->numChannels);
        const int numSamples = static_cast<int> (reader->lengthInSamples);
        juce::AudioBuffer<float> buffer (numChannels, numSamples);
        reader->read (&buffer, 0, numSamples, 0, true, true);
        sampleRate = static_cast<int> (std::lround (reader->sampleRate));
        mono.assign (static_cast<size_t> (numSamples), 0.0);
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const float* data = buffer.getReadPointer (ch);
            for (int i = 0; i < numSamples; ++i)
                mono[static_cast<size_t> (i)] += static_cast<double> (data[i]);
        }
        for (auto& v : mono) // チャンネルの平均（Python の load_mono と同じ）
            v /= numChannels;
        return true;
    }

    bool writeFloats (const juce::File& file, const std::vector<float>& data)
    {
        file.deleteFile();
        juce::FileOutputStream out (file);
        return out.openedOk() && (data.empty() || out.write (data.data(), data.size() * sizeof (float)));
    }

    double millisecondsSince (juce::int64 startTicks)
    {
        return juce::Time::highResolutionTicksToSeconds (juce::Time::getHighResolutionTicks() - startTicks) * 1000.0;
    }
}

int main (int argc, char** argv)
{
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (juce::String (juce::CharPointer_UTF8 (argv[i])));
    if (args.size() < 3)
    {
        std::cerr << "使い方: TranscribeTest <model.onnx> <in.wav> <出力フォルダ> [--threads N]" << std::endl;
        return 2;
    }
    const auto cwd = juce::File::getCurrentWorkingDirectory();
    const auto modelFile = cwd.getChildFile (args[0]);
    const auto wavFile = cwd.getChildFile (args[1]);
    const auto outDir = cwd.getChildFile (args[2]);
    int numThreads = 0;
    const int threadsAt = args.indexOf ("--threads");
    if (threadsAt >= 0 && threadsAt + 1 < args.size())
        numThreads = args[threadsAt + 1].getIntValue();

    juce::String error;
    ModelInfo info;
    if (! readModelInfo (modelFile.withFileExtension (".json"), info, error))
    {
        std::cerr << error << std::endl;
        return 1;
    }
    juce::MemoryBlock modelBytes;
    if (! modelFile.loadFileAsData (modelBytes))
    {
        std::cerr << "モデルを読めない: " << modelFile.getFullPathName() << std::endl;
        return 1;
    }
    std::vector<double> mono;
    int sampleRate = 0;
    if (! readMono (wavFile, mono, sampleRate, error))
    {
        std::cerr << error << std::endl;
        return 1;
    }
    if (! outDir.createDirectory())
    {
        std::cerr << "出力フォルダを作れない: " << outDir.getFullPathName() << std::endl;
        return 1;
    }

    try
    {
        const auto t0 = juce::Time::getHighResolutionTicks();
        const Transcriber transcriber (modelBytes.getData(), modelBytes.getSize(), info, numThreads);
        const double loadMs = millisecondsSince (t0);

        // transcribe() と同じ手順を、所要時間を測りながら 1 段ずつ行う
        auto t = juce::Time::getHighResolutionTicks();
        const auto resampled = resampleTo (mono, sampleRate, info.features.sampleRate);
        const double resampleMs = millisecondsSince (t);
        t = juce::Time::getHighResolutionTicks();
        const auto features = logMel (resampled, info.features);
        const double featuresMs = millisecondsSince (t);
        t = juce::Time::getHighResolutionTicks();
        int numFrames = 0;
        const auto probabilities = transcriber.run (features, numFrames);
        const double inferenceMs = millisecondsSince (t);
        const auto hits = pickHits (probabilities.data(), numFrames, static_cast<int> (info.classes.size()), info.features, info.thresholds);

        if (! writeFloats (outDir.getChildFile ("features.f32"), features.data) || ! writeFloats (outDir.getChildFile ("probs.f32"), probabilities))
        {
            std::cerr << "書き出せない: " << outDir.getFullPathName() << std::endl;
            return 1;
        }
        juce::String csv = "time_sec,class,probability\n";
        for (const auto& h : hits)
            csv << juce::String (h.time, 9) << "," << juce::String (info.classes[static_cast<size_t> (h.classIndex)]) << "," << juce::String (h.probability, 6) << "\n";
        outDir.getChildFile ("hits.csv").replaceWithText (csv, false, false, "\n");

        auto* obj = new juce::DynamicObject();
        obj->setProperty ("sample_rate", sampleRate);
        obj->setProperty ("seconds", static_cast<double> (mono.size()) / sampleRate);
        obj->setProperty ("frames", features.numFrames);
        obj->setProperty ("mels", features.numMels);
        obj->setProperty ("output_frames", numFrames);
        juce::Array<juce::var> classes;
        for (const auto& c : info.classes)
            classes.add (juce::String (c));
        obj->setProperty ("classes", classes);
        obj->setProperty ("hits", static_cast<int> (hits.size()));
        auto* ms = new juce::DynamicObject();
        ms->setProperty ("load_model", loadMs);
        ms->setProperty ("resample", resampleMs);
        ms->setProperty ("features", featuresMs);
        ms->setProperty ("inference", inferenceMs);
        obj->setProperty ("milliseconds", juce::var (ms));
        outDir.getChildFile ("info.json").replaceWithText (juce::JSON::toString (juce::var (obj)), false, false, "\n");

        std::cout << "音の長さ " << juce::String (static_cast<double> (mono.size()) / sampleRate, 2) << " 秒（" << sampleRate << " Hz）、フレーム " << features.numFrames
                  << "、打 " << hits.size() << "  所要: 間引き " << juce::String (resampleMs, 1) << " ms、対数メル " << juce::String (featuresMs, 1)
                  << " ms、推論 " << juce::String (inferenceMs, 1) << " ms（モデルの読み込み " << juce::String (loadMs, 1) << " ms）" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
