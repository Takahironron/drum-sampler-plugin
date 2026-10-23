#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

#include "BuiltInSounds.h"

// Phase 0 の空のプラグイン。MIDI でサンプルを鳴らすだけ。
class DrumSamplerAudioProcessor final : public juce::AudioProcessor
{
public:
    DrumSamplerAudioProcessor();
    ~DrumSamplerAudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    using AudioProcessor::processBlock;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.5; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getState() { return state; }
    const std::vector<builtin::Sound>& getSounds() const { return sounds; }

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void rebuildSounds (double sampleRate);

    juce::AudioProcessorValueTreeState state;
    std::atomic<float>* gainParam = nullptr;

    juce::Synthesiser synth;
    std::vector<builtin::Sound> sounds;
    double soundsSampleRate = 0.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumSamplerAudioProcessor)
};
