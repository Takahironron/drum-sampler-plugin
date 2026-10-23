#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace
{
    constexpr int kNumVoices = 16;
    constexpr double kDefaultSampleRate = 44100.0;
}

DrumSamplerAudioProcessor::DrumSamplerAudioProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      state (*this, nullptr, "STATE", createLayout())
{
    gainParam = state.getRawParameterValue ("gain");

    for (int i = 0; i < kNumVoices; ++i)
        synth.addVoice (new juce::SamplerVoice());

    // prepareToPlay の前に鳴らされても落ちないように、既定のサンプリング周波数で一度作っておく
    rebuildSounds (kDefaultSampleRate);
}

juce::AudioProcessorValueTreeState::ParameterLayout DrumSamplerAudioProcessor::createLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { "gain", 1 }, "Gain",
        juce::NormalisableRange<float> (-60.0f, 6.0f, 0.1f), 0.0f,
        juce::AudioParameterFloatAttributes().withLabel ("dB")));
    return layout;
}

void DrumSamplerAudioProcessor::rebuildSounds (double sampleRate)
{
    // 音を鳴らすスレッドでは呼ばない（メモリ確保をする）
    synth.clearSounds();
    sounds = builtin::makeDefaultSounds (sampleRate);
    for (const auto& s : sounds)
        synth.addSound (builtin::makeSamplerSound (s, sampleRate));
    soundsSampleRate = sampleRate;
}

void DrumSamplerAudioProcessor::prepareToPlay (double sampleRate, int)
{
    if (! juce::approximatelyEqual (sampleRate, soundsSampleRate))
        rebuildSounds (sampleRate);
    synth.setCurrentPlaybackSampleRate (sampleRate);
}

bool DrumSamplerAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
}

void DrumSamplerAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    buffer.clear();
    synth.renderNextBlock (buffer, midi, 0, buffer.getNumSamples());
    buffer.applyGain (juce::Decibels::decibelsToGain (gainParam->load()));
}

juce::AudioProcessorEditor* DrumSamplerAudioProcessor::createEditor()
{
    return new DrumSamplerAudioProcessorEditor (*this);
}

void DrumSamplerAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = state.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void DrumSamplerAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        if (xml->hasTagName (state.state.getType()))
            state.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DrumSamplerAudioProcessor();
}
