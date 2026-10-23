#include "BuiltInSounds.h"

#include <cmath>

namespace
{
    juce::AudioBuffer<float> makeKick (double sr)
    {
        const int n = (int) (0.5 * sr);
        juce::AudioBuffer<float> b (1, n);
        auto* d = b.getWritePointer (0);
        double phase = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double t = i / sr;
            const double freq = 45.0 + 110.0 * std::exp (-t * 35.0); // ピッチの落ち込み
            phase += juce::MathConstants<double>::twoPi * freq / sr;
            const double env = std::exp (-t * 7.0);
            const double click = std::exp (-t * 400.0) * 0.4;
            d[i] = (float) (std::tanh (1.6 * (std::sin (phase) * env + click)) * 0.9);
        }
        return b;
    }

    juce::AudioBuffer<float> makeSnare (double sr)
    {
        const int n = (int) (0.35 * sr);
        juce::AudioBuffer<float> b (1, n);
        auto* d = b.getWritePointer (0);
        juce::Random rng (2);
        double p1 = 0.0, p2 = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double t = i / sr;
            p1 += juce::MathConstants<double>::twoPi * 185.0 / sr;
            p2 += juce::MathConstants<double>::twoPi * 330.0 / sr;
            const double body = (std::sin (p1) * 0.6 + std::sin (p2) * 0.4) * std::exp (-t * 25.0);
            const double noise = rng.nextFloat() * 2.0 - 1.0;
            const double snares = noise * std::exp (-t * 14.0) * 0.6;
            d[i] = (float) (std::tanh (body + snares) * 0.8);
        }
        return b;
    }

    juce::AudioBuffer<float> makeHiHat (double sr)
    {
        const int n = (int) (0.12 * sr);
        juce::AudioBuffer<float> b (1, n);
        auto* d = b.getWritePointer (0);
        juce::Random rng (3);
        // 1次のハイパスで金属っぽく
        double prevIn = 0.0, prevOut = 0.0;
        const double rc = 1.0 / (juce::MathConstants<double>::twoPi * 6000.0);
        const double a = rc / (rc + 1.0 / sr);
        for (int i = 0; i < n; ++i)
        {
            const double t = i / sr;
            const double x = rng.nextFloat() * 2.0 - 1.0;
            const double y = a * (prevOut + x - prevIn);
            prevIn = x;
            prevOut = y;
            d[i] = (float) (y * std::exp (-t * 45.0) * 0.5);
        }
        return b;
    }
}

namespace builtin
{
    std::vector<Sound> makeDefaultSounds (double sampleRate)
    {
        std::vector<Sound> out;
        out.push_back ({ "Kick",  36, makeKick (sampleRate) });
        out.push_back ({ "Snare", 38, makeSnare (sampleRate) });
        out.push_back ({ "HiHat", 42, makeHiHat (sampleRate) });
        return out;
    }

    juce::SynthesiserSound::Ptr makeSamplerSound (const Sound& sound, double sampleRate)
    {
        // SamplerSound は AudioFormatReader から読むので、メモリ上のバッファを読める reader を用意する
        struct BufferReader final : public juce::AudioFormatReader
        {
            BufferReader (const juce::AudioBuffer<float>& b, double sr)
                : juce::AudioFormatReader (nullptr, "memory"), buffer (b)
            {
                sampleRate = sr;
                bitsPerSample = 32;
                lengthInSamples = b.getNumSamples();
                numChannels = (unsigned int) b.getNumChannels();
                usesFloatingPointData = true;
            }

            bool readSamples (int* const* destChannels, int numDestChannels, int startOffsetInDestBuffer,
                              juce::int64 startSampleInFile, int numSamples) override
            {
                clearSamplesBeyondAvailableLength (destChannels, numDestChannels, startOffsetInDestBuffer,
                                                  startSampleInFile, numSamples, lengthInSamples);
                if (numSamples <= 0)
                    return true;

                for (int ch = 0; ch < numDestChannels; ++ch)
                {
                    if (destChannels[ch] == nullptr)
                        continue;

                    auto* dest = reinterpret_cast<float*> (destChannels[ch]) + startOffsetInDestBuffer;
                    const int srcCh = juce::jmin (ch, buffer.getNumChannels() - 1);
                    const auto* src = buffer.getReadPointer (srcCh) + (int) startSampleInFile;
                    std::copy (src, src + numSamples, dest);
                }
                return true;
            }

            const juce::AudioBuffer<float>& buffer;
        };

        BufferReader reader (sound.buffer, sampleRate);
        juce::BigInteger notes;
        notes.setBit (sound.midiNote);

        const double maxLengthSec = sound.buffer.getNumSamples() / sampleRate + 0.1;
        return new juce::SamplerSound (sound.name, reader, notes, sound.midiNote,
                                       0.001, 0.05, maxLengthSec);
    }
}
