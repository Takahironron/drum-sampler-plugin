#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

#include "KitModel.h"
#include "SampleEngine.h"

// Phase 1 の最小版: kit.json（Python の解析が作ったキット）を読んで MIDI で鳴らす。
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
    double getTailLengthSeconds() const override { return 2.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getState() { return state; }

    // メッセージスレッドから。kit.json を読んでエンジンに渡す。失敗したら false と理由
    bool loadKit (const juce::File& kitJson, juce::String& error);
    void useBuiltInKit();
    std::shared_ptr<const kit::KitModel> currentKit() const { return kitForDisplay; }
    juce::File currentKitFile() const { return kitFile; }

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    juce::AudioProcessorValueTreeState state;
    std::atomic<float>* gainParam = nullptr;

    kit::SampleEngine engine;
    std::shared_ptr<const kit::KitModel> kitForDisplay; // メッセージスレッドが持つ参照（表示と解放の責任）
    std::shared_ptr<const kit::KitModel> previousKit;   // 差し替え後も音を鳴らすスレッドが使い終わるまで保持
    juce::File kitFile;
    double currentSampleRate = 44100.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumSamplerAudioProcessor)
};
