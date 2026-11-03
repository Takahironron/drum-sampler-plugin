#include "SampleEngine.h"

#include <cmath>

namespace kit
{
    std::atomic<juce::int64> SampleVoice::counter { 0 };

    void SampleVoice::start (const Sample* s, double hostRate, float g, int part) noexcept
    {
        sample = s;
        position = 0.0;
        increment = s->sampleRate / hostRate;
        gain = g;
        fade = 1.0f;
        fadeStep = 0.0f;
        ownerPart = part;
        startOrder = ++counter;
    }

    void SampleVoice::stop (double hostRate) noexcept
    {
        if (sample != nullptr && fadeStep <= 0.0f)
            fadeStep = 1.0f / (float) juce::jmax (1.0, 0.005 * hostRate);
    }

    void SampleVoice::render (juce::AudioBuffer<float>& out, int startSample, int numSamples) noexcept
    {
        if (sample == nullptr)
            return;
        const auto* src = sample->buffer.getReadPointer (0);
        const int n = sample->buffer.getNumSamples();
        const int numCh = out.getNumChannels();
        for (int i = 0; i < numSamples; ++i)
        {
            const int idx = (int) position;
            if (idx + 1 >= n || fade <= 0.0f)
            {
                sample = nullptr;
                return;
            }
            const float frac = (float) (position - idx);
            const float v = (src[idx] + (src[idx + 1] - src[idx]) * frac) * gain * fade;
            for (int ch = 0; ch < numCh; ++ch)
                out.addSample (ch, startSample + i, v);
            position += increment;
            if (fadeStep > 0.0f)
                fade -= fadeStep;
        }
    }

    void SampleEngine::setKit (std::shared_ptr<const KitModel> newKit)
    {
        const juce::SpinLock::ScopedLockType lock (pendingLock);
        pendingKit = std::move (newKit);
    }

    std::shared_ptr<const KitModel> SampleEngine::currentKit() const
    {
        return kit;
    }

    void SampleEngine::beginBlock() noexcept
    {
        const juce::SpinLock::ScopedTryLockType lock (pendingLock);
        if (lock.isLocked() && pendingKit != nullptr)
        {
            // 古いキットの解放（shared_ptr の減算）は音を鳴らすスレッドで起きうるが、最後の参照はメッセージスレッド側が持つ設計
            kit = pendingKit;
            pendingKit = nullptr;
            for (auto& v : voices)
                v.stop (sampleRate);
        }
    }

    void SampleEngine::noteOn (int midiNote, float velocity01) noexcept
    {
        if (kit == nullptr)
            return;
        const Part* part = kit->partForNote (midiNote);
        if (part == nullptr || part->samples.empty())
            return;
        const int partIndex = (int) (part - kit->parts().data());

        // 強さ → 目標のピーク dB（キットの中の最小〜最大を線形に）。その近くの音を選び、残りの差を音量で合わせる
        const float targetDb = part->minPeakDb + (part->maxPeakDb - part->minPeakDb) * velocity01;
        const int n = (int) part->samples.size();
        const int zones = juce::jmin (4, n);
        const int zone = juce::jlimit (0, zones - 1, (int) (velocity01 * (float) zones));
        const int zoneStart = zone * n / zones;
        const int zoneEnd = juce::jmax (zoneStart + 1, (zone + 1) * n / zones);
        auto& rr = roundRobin[(size_t) juce::jlimit (0, 15, partIndex)];
        const int pick = zoneStart + (rr++ % (zoneEnd - zoneStart));
        const Sample& s = part->samples[(size_t) pick];
        const float gainDb = juce::jlimit (-12.0f, 12.0f, targetDb - s.peakDb);
        const float gain = juce::Decibels::decibelsToGain (gainDb) * (0.35f + 0.65f * velocity01);

        if (part->choke)
            for (auto& v : voices)
                if (v.isActive() && v.owner() == partIndex)
                    v.stop (sampleRate);

        SampleVoice* target = nullptr;
        for (auto& v : voices)
            if (! v.isActive()) { target = &v; break; }
        if (target == nullptr)
        {
            target = &voices[0];
            for (auto& v : voices)
                if (v.startedAt() < target->startedAt())
                    target = &v;
        }
        target->start (&s, sampleRate, gain, partIndex);
    }

    void SampleEngine::render (juce::AudioBuffer<float>& out, int startSample, int numSamples) noexcept
    {
        for (auto& v : voices)
            v.render (out, startSample, numSamples);
    }
}
