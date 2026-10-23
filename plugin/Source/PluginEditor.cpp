#include "PluginEditor.h"

DrumSamplerAudioProcessorEditor::DrumSamplerAudioProcessorEditor (DrumSamplerAudioProcessor& p)
    : AudioProcessorEditor (&p), processor (p)
{
    gainSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    gainSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
    addAndMakeVisible (gainSlider);
    gainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
        processor.getState(), "gain", gainSlider);

    gainLabel.setText ("Gain", juce::dontSendNotification);
    gainLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (gainLabel);

    setSize (360, 200);
}

void DrumSamplerAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    g.setColour (juce::Colours::white);
    g.setFont (juce::FontOptions (16.0f));
    g.drawText ("DrumSamplerPlugin (Phase 0)", getLocalBounds().removeFromTop (30), juce::Justification::centred);

    g.setFont (juce::FontOptions (13.0f));
    juce::String lines;
    for (const auto& s : processor.getSounds())
        lines << "MIDI " << s.midiNote << " : " << s.name << "\n";
    g.drawMultiLineText (lines, 20, 60, 160);
}

void DrumSamplerAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced (20);
    area.removeFromTop (20);
    auto right = area.removeFromRight (140);
    gainLabel.setBounds (right.removeFromTop (20));
    gainSlider.setBounds (right.removeFromTop (110));
}
