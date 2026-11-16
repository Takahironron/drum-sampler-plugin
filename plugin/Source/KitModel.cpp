#include "KitModel.h"

#include "BuiltInSounds.h"

#include <algorithm>
#include <cmath>

namespace kit
{
    const char* KitModel::classForNote (int n) noexcept
    {
        switch (n)
        {
            case 35: case 36: return "kick";
            case 37: case 38: case 39: case 40: return "snare";
            case 42: case 44: case 46: case 22: case 26: return "hihat";
            case 41: case 43: case 45: case 47: case 48: case 50: case 58: return "tom";
            case 49: case 51: case 52: case 53: case 55: case 57: case 59: return "cymbal";
            default: return nullptr;
        }
    }

    int KitModel::noteForClass (const juce::String& cls) noexcept
    {
        if (cls == "kick") return 36;
        if (cls == "snare") return 38;
        if (cls == "hihat") return 42;
        if (cls == "tom") return 45;
        if (cls == "cymbal") return 49;
        return 60;
    }

    const Part* KitModel::partForNote (int midiNote) const noexcept
    {
        const char* cls = classForNote (midiNote);
        if (cls == nullptr)
            return nullptr;
        for (const auto& p : parts_)
            if (p.cls == cls)
                return &p;
        return nullptr;
    }

    double KitModel::levelForVelocity (int velocity, double loDb, double hiDb) noexcept
    {
        return loDb + (hiDb - loDb) * (double) (juce::jlimit (1, 127, velocity) - 1) / 126.0;
    }

    static void finishPart (Part& part)
    {
        // 同じピークの音の順を kit.json の順のままにする（Python の参照実装と同じ音を選ぶため）
        std::stable_sort (part.samples.begin(), part.samples.end(), [] (const Sample& a, const Sample& b) { return a.peakDb < b.peakDb; });
        part.minPeakDb = part.samples.empty() ? 0.0 : part.samples.front().peakDb;
        part.maxPeakDb = part.samples.empty() ? 0.0 : part.samples.back().peakDb;
        // 強さの範囲の既定: 最も大きい音が 127、そこから 30 dB 下（か最も小さい音）が 1。kit.json に level_range_db があれば後で上書きする
        part.levelHiDb = part.maxPeakDb;
        part.levelLoDb = juce::jmin (part.minPeakDb, part.maxPeakDb - KitModel::velocityRangeDb);
        part.midiNote = KitModel::noteForClass (part.cls);
        part.choke = (part.cls != "cymbal");
    }

    bool KitModel::loadFromJson (const juce::File& kitJson, juce::String& error)
    {
        const auto parsed = juce::JSON::parse (kitJson);
        if (! parsed.isObject())
        {
            error = "kit.json を読めない: " + kitJson.getFullPathName();
            return false;
        }
        const auto* obj = parsed.getDynamicObject();
        const auto partsVar = obj->getProperty ("parts");
        if (! partsVar.isObject())
        {
            error = "kit.json に parts が無い";
            return false;
        }

        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        const auto dir = kitJson.getParentDirectory();
        const auto ranges = obj->getProperty ("level_range_db");
        std::vector<Part> parts;

        for (const auto& kv : partsVar.getDynamicObject()->getProperties())
        {
            Part part;
            part.cls = kv.name.toString();
            const auto layers = kv.value;
            if (! layers.isArray())
                continue;
            for (const auto& layerVar : *layers.getArray())
            {
                if (! layerVar.isObject())
                    continue;
                const auto files = layerVar.getProperty ("files", juce::var());
                const auto peaks = layerVar.getProperty ("peaks_db", juce::var());
                if (! files.isArray())
                    continue;
                for (int i = 0; i < files.getArray()->size(); ++i)
                {
                    const auto rel = files[i].toString();
                    const auto file = dir.getChildFile (rel);
                    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));
                    if (reader == nullptr)
                    {
                        error = "WAV を読めない: " + file.getFullPathName();
                        return false;
                    }
                    Sample s;
                    s.file = rel;
                    s.sampleRate = reader->sampleRate;
                    const int n = (int) reader->lengthInSamples;
                    juce::AudioBuffer<float> tmp ((int) reader->numChannels, n);
                    reader->read (&tmp, 0, n, 0, true, true);
                    s.buffer.setSize (1, n);
                    s.buffer.clear();
                    for (int ch = 0; ch < tmp.getNumChannels(); ++ch)
                        s.buffer.addFrom (0, 0, tmp, ch, 0, n, 1.0f / (float) tmp.getNumChannels());
                    if (peaks.isArray() && i < peaks.getArray()->size())
                        s.peakDb = (double) peaks[i];
                    else
                        s.peakDb = 20.0 * std::log10 ((double) s.buffer.getMagnitude (0, n) + 1.0e-9);
                    part.samples.push_back (std::move (s));
                }
            }
            if (! part.samples.empty())
            {
                finishPart (part);
                // 強さ 1〜127 が対応する打のピークの範囲 [下限, 上限]（解析が、元の曲の打の音量から決めたもの）
                const auto range = ranges.isObject() ? ranges.getProperty (kv.name, juce::var()) : juce::var();
                if (range.isArray() && range.getArray()->size() == 2 && (double) range[1] > (double) range[0])
                {
                    part.levelLoDb = (double) range[0];
                    part.levelHiDb = (double) range[1];
                }
                parts.push_back (std::move (part));
            }
        }
        if (parts.empty())
        {
            error = "kit.json に音が無い";
            return false;
        }
        parts_ = std::move (parts);
        name_ = kitJson.getParentDirectory().getFileName();
        // 各音は打点のこの秒数だけ前から始まる。書いていない kit.json（2026-11-16 より前）も同じ長さで切り出してある
        const auto lead = obj->getProperty ("lead_sec");
        leadSec_ = (lead.isDouble() || lead.isInt()) ? juce::jlimit (0.0, 0.05, (double) lead) : defaultLeadSeconds;
        return true;
    }

    std::shared_ptr<const KitModel> KitModel::makeBuiltIn (double sampleRate)
    {
        auto model = std::make_shared<KitModel>();
        for (const auto& snd : builtin::makeDefaultSounds (sampleRate))
        {
            Part part;
            part.cls = KitModel::classForNote (snd.midiNote);
            Sample s;
            s.buffer = snd.buffer;
            s.sampleRate = sampleRate;
            s.peakDb = 20.0 * std::log10 ((double) s.buffer.getMagnitude (0, s.buffer.getNumSamples()) + 1.0e-9);
            s.file = snd.name;
            part.samples.push_back (std::move (s));
            finishPart (part);
            model->parts_.push_back (std::move (part));
        }
        model->name_ = "Built-in";
        model->leadSec_ = 0.0; // 合成の音は頭から立ち上がる
        return model;
    }
}
