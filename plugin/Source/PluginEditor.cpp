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

    loadButton.onClick = [this] { chooseKit(); };
    addAndMakeVisible (loadButton);
    builtInButton.onClick = [this] { processor.useBuiltInKit(); refreshKitInfo(); };
    addAndMakeVisible (builtInButton);

    kitInfo.setJustificationType (juce::Justification::topLeft);
    kitInfo.setFont (juce::FontOptions (13.0f));
    addAndMakeVisible (kitInfo);
    refreshKitInfo();

    setSize (460, 260);
}

void DrumSamplerAudioProcessorEditor::chooseKit()
{
    chooser = std::make_unique<juce::FileChooser> ("kit.json を選ぶ", processor.currentKitFile().getParentDirectory(), "*.json");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this] (const juce::FileChooser& fc)
                          {
                              const auto file = fc.getResult();
                              if (file == juce::File())
                                  return;
                              juce::String error;
                              if (! processor.loadKit (file, error))
                                  kitInfo.setText ("読み込み失敗: " + error, juce::dontSendNotification);
                              else
                                  refreshKitInfo();
                          });
}

void DrumSamplerAudioProcessorEditor::refreshKitInfo()
{
    juce::String text;
    if (auto kit = processor.currentKit())
    {
        text << "キット: " << kit->name() << "\n";
        for (const auto& part : kit->parts())
            text << "  MIDI " << part.midiNote << "  " << part.cls << "  " << (int) part.samples.size() << " 音 ("
                 << juce::String (part.minPeakDb, 1) << " ～ " << juce::String (part.maxPeakDb, 1) << " dB)"
                 << (part.choke ? "  チョーク" : "") << "\n";
    }
    kitInfo.setText (text, juce::dontSendNotification);
}

void DrumSamplerAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    g.setColour (juce::Colours::white);
    g.setFont (juce::FontOptions (16.0f));
    g.drawText ("DrumSamplerPlugin (Phase 1)", getLocalBounds().removeFromTop (30), juce::Justification::centred);
}

void DrumSamplerAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced (16);
    area.removeFromTop (24);
    auto right = area.removeFromRight (140);
    gainLabel.setBounds (right.removeFromTop (20));
    gainSlider.setBounds (right.removeFromTop (110));
    auto buttons = area.removeFromTop (28);
    loadButton.setBounds (buttons.removeFromLeft (150));
    buttons.removeFromLeft (8);
    builtInButton.setBounds (buttons.removeFromLeft (120));
    area.removeFromTop (8);
    kitInfo.setBounds (area);
}
