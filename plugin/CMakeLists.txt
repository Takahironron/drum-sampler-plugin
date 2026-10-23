# Phase 0 の空のプラグイン: MIDI でサンプルを鳴らすだけ（VST3 と Standalone）。
#
# 使い方（Windows、Visual Studio 2022）:
#   cmake -S plugin -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
#   → build/DrumSamplerPlugin_artefacts/Release/VST3/DrumSamplerPlugin.vst3
#
# JUCE は FetchContent で取る。手元の JUCE を使うなら -DJUCE_SOURCE_DIR=<JUCE のフォルダ>。

cmake_minimum_required(VERSION 3.22)

project(DrumSamplerPlugin VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(JUCE_GIT_TAG "8.0.15" CACHE STRING "取ってくる JUCE のタグ")
set(JUCE_SOURCE_DIR "" CACHE PATH "手元の JUCE を使うときのフォルダ（空なら FetchContent）")

if(JUCE_SOURCE_DIR)
    add_subdirectory("${JUCE_SOURCE_DIR}" JUCE)
else()
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        ${JUCE_GIT_TAG}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
    FetchContent_MakeAvailable(JUCE)
endif()

juce_add_plugin(DrumSamplerPlugin
    PRODUCT_NAME                "DrumSamplerPlugin"
    COMPANY_NAME                "Takahironron"
    BUNDLE_ID                   "com.takahironron.drumsamplerplugin"
    PLUGIN_MANUFACTURER_CODE    Tkhr
    PLUGIN_CODE                 Dsp0
    FORMATS                     VST3 Standalone
    IS_SYNTH                    TRUE
    NEEDS_MIDI_INPUT            TRUE
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD     FALSE
    VST3_CATEGORIES             Instrument Drum Sampler
)

target_sources(DrumSamplerPlugin PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    Source/BuiltInSounds.cpp
    Source/BuiltInSounds.h
)

target_compile_definitions(DrumSamplerPlugin PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(DrumSamplerPlugin PRIVATE
    juce::juce_audio_utils
  PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)
