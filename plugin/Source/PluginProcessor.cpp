#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <cmath>

DrumSamplerAudioProcessor::DrumSamplerAudioProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      state (*this, nullptr, "STATE", createLayout())
{
    gainParam = state.getRawParameterValue ("gain");
    useBuiltInKit();
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

void DrumSamplerAudioProcessor::useBuiltInKit()
{
    auto model = kit::KitModel::makeBuiltIn (currentSampleRate);
    previousKit = kitForDisplay;
    kitForDisplay = model;
    kitFile = juce::File();
    engine.setKit (model);
    state.state.setProperty ("kitPath", "", nullptr);
    updateLatency();
}

void DrumSamplerAudioProcessor::updateLatency()
{
    // キットの音は、立ち上がりの少し前（leadSeconds）から始まる。その長さを遅れとして伝えると、ホストが補正して、
    // 立ち上がりが MIDI の打点（と、並べて置いた元のステム）に重なる
    const auto model = kitForDisplay;
    setLatencySamples (model != nullptr ? (int) std::lround (model->leadSeconds() * currentSampleRate) : 0);
}

bool DrumSamplerAudioProcessor::loadKit (const juce::File& kitJson, juce::String& error)
{
    auto model = std::make_shared<kit::KitModel>();
    if (! model->loadFromJson (kitJson, error))
        return false;
    previousKit = kitForDisplay;
    kitForDisplay = model;
    kitFile = kitJson;
    engine.setKit (model);
    state.state.setProperty ("kitPath", kitJson.getFullPathName(), nullptr);
    updateLatency();
    return true;
}

void DrumSamplerAudioProcessor::prepareToPlay (double sampleRate, int)
{
    currentSampleRate = sampleRate;
    engine.prepare (sampleRate);
    if (kitFile == juce::File() && kitForDisplay != nullptr && kitForDisplay->name() == "Built-in")
        useBuiltInKit(); // 合成の仮の音はサンプリング周波数に合わせて作り直す
    updateLatency();
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
    engine.beginBlock();

    int pos = 0;
    const int total = buffer.getNumSamples();
    for (const auto meta : midi)
    {
        const auto msg = meta.getMessage();
        const int at = juce::jlimit (0, total, meta.samplePosition);
        if (at > pos)
        {
            engine.render (buffer, pos, at - pos);
            pos = at;
        }
        if (msg.isNoteOn())
            engine.noteOn (msg.getNoteNumber(), (int) msg.getVelocity());
    }
    if (total > pos)
        engine.render (buffer, pos, total - pos);

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
    {
        if (xml->hasTagName (state.state.getType()))
        {
            state.replaceState (juce::ValueTree::fromXml (*xml));
            const auto path = state.state.getProperty ("kitPath").toString();
            if (path.isNotEmpty())
            {
                juce::String error;
                const juce::File f (path);
                if (! (f.existsAsFile() && loadKit (f, error)))
                    useBuiltInKit();
            }
        }
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DrumSamplerAudioProcessor();
}
