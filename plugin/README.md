# plugin — JUCE のプラグイン（Phase 1 の最小版）

Python の解析が作ったキット（`kit.json` と代表音の WAV）を読み、MIDI で鳴らす VST3／Standalone。Phase 0 では「コード合成の 3 音を鳴らすだけの空のプラグイン」で、Phase 1 で `KitModel` と自前のボイス管理を足した。

## 中身

| ファイル | 役目 |
|---|---|
| `Source/KitModel.h/.cpp` | キットのデータ（計画書のモジュール `KitModel`）。`kit.json` を読み、WAV をメモリに載せる。種類（kick／snare／hihat／tom／cymbal）→ GM ノート（36／38／42／45／49。周辺のノートも同じ種類に寄せる）。読み込みはメッセージスレッド |
| `Source/SampleEngine.h/.cpp` | 音を鳴らすエンジン（`Engine` の最小版）。32 ボイスの固定プール、強さ → ピークの目標 dB → 近い音域（最大 4 段）からラウンドロビンで選び、残りの差を音量で合わせる（±12 dB）。ハイハットはチョーク（5 ms で止める）。キットの差し替えはロックの試行で受け渡し、音を鳴らすスレッドではメモリ確保・ロック待ち・ファイル入出力をしない |
| `Source/BuiltInSounds.*` | コード合成の仮の音（`kit.json` が無いときの既定のキット） |
| `Source/PluginProcessor.*`、`PluginEditor.*` | パラメータ（Gain）、状態の保存（キットのパスを含む）、画面（「kit.json を読む」「仮の音に戻す」、キットの内訳の表示） |
| `Tools/KitRenderTest.cpp` | DAW 無しでエンジンを通す確認用のコンソールプログラム（`kit.json` → 8 ビートの MIDI → WAV）。CI では Python の `make_demo_kit.py` が作ったキットで毎回通す |

## ビルド（Windows、Visual Studio 2022）

```
cmake -S plugin -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target DrumSamplerPlugin_VST3 DrumSamplerPlugin_Standalone KitRenderTest
```

出力: `build/DrumSamplerPlugin_artefacts/Release/VST3/DrumSamplerPlugin.vst3`、`.../Standalone/DrumSamplerPlugin.exe`、`build/KitRenderTest_artefacts/Release/KitRenderTest.exe`

ビルドせずに試すなら、GitHub Actions の `build` の成果物 `DrumSamplerPlugin-windows-standalone`（exe）と `DrumSamplerPlugin-windows-vst3` を使う。

JUCE は CMake の FetchContent で `8.0.15` を取る。手元の JUCE を使うなら `-DJUCE_SOURCE_DIR=<JUCE のフォルダ>`。

## キットを鳴らす

1. 実ステムからキットを作る（方式 C: 特徴で選んだ代表音 24 音／種類）。

   ```
   python analysis\run_eval.py data\stems --model analysis\models\crnn_egmd_v5.onnx --only a07 --save-kit kit
   ```

   `data\stems\a07\kit\kit.json` と `kick\`、`snare\`… の WAV ができる。合成音のデモなら `python analysis\make_demo_kit.py data\kits\demo`。

2. Standalone（か DAW の VST3）で「kit.json を読む」からその `kit.json` を選ぶ。画面に種類ごとの音の数とピークの範囲が出る。
3. MIDI の 36（キック）、38（スネア）、42（ハイハット）、45（タム）、49（シンバル）で鳴る。強さで音が変わる。

DAW 無しで確かめるなら:

```
build\KitRenderTest_artefacts\Release\KitRenderTest.exe data\stems\a07\kit\kit.json out.wav 8
```

## pluginval

```
pluginval --strictness-level 5 --validate-in-process --validate build/DrumSamplerPlugin_artefacts/Release/VST3/DrumSamplerPlugin.vst3
```

GitHub Actions（`.github/workflows/build.yml`）で Windows は pluginval、Linux は `KitRenderTest` の通し、macOS はコンパイルの確認。Linux では画面のテストを除いた pluginval（`--skip-gui-tests`）が SUCCESS（2026-11-03）。

## ライセンスの注意

- JUCE は AGPLv3 と商用ライセンスの二本立て。JUCE 8 の Starter（年 $20,000 まで）はスプラッシュ画面不要。販売前の条件は `docs/phase0/ライセンス一覧.md`。
- VST3 SDK は JUCE に同梱。JUCE 8.0.15 の同梱版は 3.8.0 で MIT。VST の名称・ロゴを使うなら Steinberg の商標ガイドラインに従う。

## 次の段階

解析（対数メル、CRNN の推論、NMFD、切り出し、代表音の選択）を C++／ONNX Runtime に移す（`docs/plan/Phase1作業順.md`）。
