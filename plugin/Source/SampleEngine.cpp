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
        chokeCountdown = -1;
        fadePos = 0;
        ownerPart = part;
        startOrder = ++counter;
    }

    void SampleVoice::choke (int delaySamples, int fadeSamples) noexcept
    {
        if (sample == nullptr || chokeCountdown >= 0)
            return;
        chokeCountdown = juce::jmax (0, delaySamples);
        fadePos = 0;
        fadeLength = juce::jmax (1, fadeSamples);
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
            if (idx >= n)
            {
                sample = nullptr;
                return;
            }
            float g = gain;
            if (chokeCountdown > 0)
            {
                --chokeCountdown;
            }
            else if (chokeCountdown == 0)
            {
                // フェード: 1 から始まり、fadeLength サンプル目で 0（そこで止める）
                if (fadePos >= fadeLength)
                {
                    sample = nullptr;
                    return;
                }
                g *= 1.0f - (float) fadePos / (float) fadeLength;
                ++fadePos;
            }
            const float frac = (float) (position - idx);
            const float next = idx + 1 < n ? src[idx + 1] : 0.0f;
            const float v = (src[idx] + (next - src[idx]) * frac) * g;
            for (int ch = 0; ch < numCh; ++ch)
                out.addSample (ch, startSample + i, v);
            position += increment;
        }
    }

    void SampleEngine::prepare (double hostRate) noexcept
    {
        sampleRate = hostRate;
        updateTimings();
    }

    void SampleEngine::updateTimings() noexcept
    {
        leadSamples = kit != nullptr ? (int) std::lround (kit->leadSeconds() * sampleRate) : 0;
        fadeSamples = juce::jmax (1, (int) (chokeFadeSeconds * sampleRate));
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
            updateTimings();
            for (auto& v : voices)
                v.choke (0, fadeSamples);
        }
    }

    void SampleEngine::noteOn (int midiNote, int velocity) noexcept
    {
        if (kit == nullptr)
            return;
        const Part* part = kit->partForNote (midiNote);
        if (part == nullptr || part->samples.empty())
            return;
        const int partIndex = (int) (part - kit->parts().data());

        // 強さ → 目標のピーク dB。ピークが目標に最も近い音を選び（同じ近さなら小さい方）、残りの差を音量で合わせる
        const double targetDb = KitModel::levelForVelocity (velocity, part->levelLoDb, part->levelHiDb);
        size_t pick = 0;
        double best = std::abs (part->samples[0].peakDb - targetDb);
        for (size_t i = 1; i < part->samples.size(); ++i)
        {
            const double d = std::abs (part->samples[i].peakDb - targetDb);
            if (d < best)
            {
                best = d;
                pick = i;
            }
        }
        const Sample& s = part->samples[pick];
        const float gain = (float) std::pow (10.0, juce::jmin (targetDb - s.peakDb, maxBoostDb) / 20.0);

        if (part->choke)
            for (auto& v : voices)
                if (v.isActive() && v.owner() == partIndex)
                    v.choke (leadSamples, fadeSamples);

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
