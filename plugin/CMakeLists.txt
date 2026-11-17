# Phase 1 の最小版: Python が作った kit.json を読んで MIDI で鳴らす（VST3 と Standalone）。Phase 0 では空のプラグインだった。
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
    Source/KitModel.cpp
    Source/KitModel.h
    Source/SampleEngine.cpp
    Source/SampleEngine.h
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

# DAW 無しでエンジンを通す確認用のコンソールプログラム（kit.json → WAV）
juce_add_console_app(KitRenderTest PRODUCT_NAME "KitRenderTest")
target_sources(KitRenderTest PRIVATE
    Tools/KitRenderTest.cpp
    Source/KitModel.cpp
    Source/SampleEngine.cpp
    Source/BuiltInSounds.cpp
)
target_compile_definitions(KitRenderTest PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_MODAL_LOOPS_PERMITTED=0)
target_link_libraries(KitRenderTest PRIVATE juce::juce_audio_utils PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)

# 採譜（ONNX Runtime）を通す確認用のコンソールプログラム（WAV → 対数メル → 推論 → 打点）。Python の参照実装と突き合わせる
# （analysis/transcribe_parity.py）。ONNX Runtime（MIT）は配布物を取ってくる。手元のものを使うなら -DONNXRUNTIME_ROOT=<フォルダ>。
#   cmake -S plugin -B build -DDRUM_WITH_ONNXRUNTIME=ON ...
#   cmake --build build --target TranscribeTest
option(DRUM_WITH_ONNXRUNTIME "採譜（ONNX Runtime）を使うプログラムをビルドする" OFF)
set(ONNXRUNTIME_VERSION "1.31.0" CACHE STRING "取ってくる ONNX Runtime の版（Python 側の onnxruntime と同じにする）")
set(ONNXRUNTIME_ROOT "" CACHE PATH "手元の ONNX Runtime（include と lib のあるフォルダ）。空なら配布物を取ってくる")

if(DRUM_WITH_ONNXRUNTIME)
    if(NOT ONNXRUNTIME_ROOT)
        if(WIN32)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
                set(_ort_package "onnxruntime-win-arm64-${ONNXRUNTIME_VERSION}.zip")
            else()
                set(_ort_package "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
            endif()
        elseif(APPLE)
            set(_ort_package "onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz") # 1.31.0 の配布物は arm64 だけ
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(_ort_package "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
        else()
            set(_ort_package "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
        endif()
        include(FetchContent)
        FetchContent_Declare(
            onnxruntime_prebuilt
            URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${_ort_package}"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        set(ONNXRUNTIME_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}" CACHE PATH "" FORCE)
    endif()
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")

    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")
    if(WIN32)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_IMPLIB   "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
            IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
    elseif(APPLE)
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
    else()
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()

    juce_add_console_app(TranscribeTest PRODUCT_NAME "TranscribeTest")
    target_sources(TranscribeTest PRIVATE
        Tools/TranscribeTest.cpp
        Source/Analysis/Features.cpp
        Source/Analysis/Features.h
        Source/Analysis/Transcriber.cpp
        Source/Analysis/Transcriber.h
    )
    target_compile_definitions(TranscribeTest PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_MODAL_LOOPS_PERMITTED=0)
    target_link_libraries(TranscribeTest PRIVATE juce::juce_audio_utils onnxruntime PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)
    if(WIN32) # 実行ファイルの隣に DLL を置く
        add_custom_command(TARGET TranscribeTest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:TranscribeTest>"
            VERBATIM)
    endif()
endif()
