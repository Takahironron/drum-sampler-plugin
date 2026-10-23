# plugin — JUCE の空のプラグイン（Phase 0）

MIDI でサンプルを鳴らすだけの VST3／Standalone。Phase 0 の目的は「Windows でビルドでき、pluginval を通る」ことの確認（計画書 9章 4）。

- 音声ファイルはコミットしないので、キック・スネア・ハイハット風の3音を起動時にコードで合成する（`Source/BuiltInSounds.cpp`）。GM ドラムマップの 36／38／42 で鳴る。
- パラメータは Gain（dB）だけ。状態の保存・復元は `AudioProcessorValueTreeState`。
- 実時間の決まり（CLAUDE.md）: 音を鳴らすスレッド（`processBlock`）ではメモリ確保・ロック・ファイル入出力・推論をしない。音の合成と `SamplerSound` の作成はコンストラクタと `prepareToPlay` で行う。

## ビルド（Windows、Visual Studio 2022）

```
cmake -S plugin -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target DrumSamplerPlugin_VST3 DrumSamplerPlugin_Standalone
```

出力: `build/DrumSamplerPlugin_artefacts/Release/VST3/DrumSamplerPlugin.vst3`、`.../Standalone/DrumSamplerPlugin.exe`

JUCE は CMake の FetchContent で `8.0.15` を取る。手元の JUCE を使うなら `-DJUCE_SOURCE_DIR=<JUCE のフォルダ>`。

## pluginval

```
pluginval --strictness-level 5 --validate-in-process --validate build/DrumSamplerPlugin_artefacts/Release/VST3/DrumSamplerPlugin.vst3
```

GitHub Actions（`.github/workflows/build.yml`）でも Windows で同じ検査を走らせる。Linux と macOS はコンパイルの確認だけ。

## Linux でのビルド（開発者の確認用）

```
sudo apt-get install -y ninja-build libasound2-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libfreetype6-dev libfontconfig1-dev libgl1-mesa-dev libcurl4-openssl-dev
cmake -S plugin -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target DrumSamplerPlugin_VST3 DrumSamplerPlugin_Standalone
```

## ライセンスの注意

- JUCE は AGPLv3 と商用ライセンスの二本立て。JUCE 8 ではスプラッシュ画面のフラグ（`JUCE_DISPLAY_SPLASH_SCREEN`）は使われない（ビルド時の JUCE の警告で確認）。販売前の条件は `docs/phase0/ライセンス一覧.md`。
- VST3 SDK は JUCE に同梱される。JUCE 8.0.15 の同梱版は 3.8.0 で MIT（2026-10-23 に `LICENSE.txt` を確認）。閉じたソースの配布に契約の署名は不要。VST の名称・ロゴを使うなら Steinberg の商標ガイドラインに従う。
- GitHub Actions で Windows（windows-2022、VS 2022）・Linux・macOS のビルドが通り、Windows で pluginval（strictness 5）が SUCCESS（2026-10-23）。Takahiro の手元での確認は未。

## 次の段階（Phase 1）へ向けて

`Engine` モジュール（`SampleVoice`・チョーク・FX チェーン）はこのプロジェクトの上に足す。`juce::Synthesiser` と `SamplerVoice` は Phase 0 の確認用で、Phase 1 で自前のボイス管理に置き換える予定。
