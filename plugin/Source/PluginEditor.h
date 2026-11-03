#pragma once

#include "PluginProcessor.h"

class DrumSamplerAudioProcessorEditor final : public juce::AudioProcessorEditor
{
public:
    explicit DrumSamplerAudioProcessorEditor (DrumSamplerAudioProcessor&);
    ~DrumSamplerAudioProcessorEditor() override = default;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void chooseKit();
    void refreshKitInfo();

    DrumSamplerAudioProcessor& processor;
    juce::Slider gainSlider;
    juce::Label gainLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> gainAttachment;
    juce::TextButton loadButton { "kit.json を読む" };
    juce::TextButton builtInButton { "仮の音に戻す" };
    juce::Label kitInfo;
    std::unique_ptr<juce::FileChooser> chooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumSamplerAudioProcessorEditor)
};
