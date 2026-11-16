# plugin — JUCE のプラグイン（Phase 1 の最小版）

Python の解析が作ったキット（`kit.json` と代表音の WAV）を読み、MIDI で鳴らす VST3／Standalone。Phase 0 では「コード合成の 3 音を鳴らすだけの空のプラグイン」で、Phase 1 で `KitModel` と自前のボイス管理を足した。

## 中身

| ファイル | 役目 |
|---|---|
| `Source/KitModel.h/.cpp` | キットのデータ（計画書のモジュール `KitModel`）。`kit.json` を読み、WAV をメモリに載せる。種類（kick／snare／hihat／tom／cymbal）→ GM ノート（36／38／42／45／49。周辺のノートも同じ種類に寄せる）。読み込みはメッセージスレッド |
| `Source/SampleEngine.h/.cpp` | 音を鳴らすエンジン（`Engine` の最小版）。32 ボイスの固定プール。**鳴らし方は Python の参照実装（`analysis/drumeval/extract.py` の `render_midi`）と同じ**: 強さ 1〜127 → 目標のピーク dB（`kit.json` の `level_range_db` の範囲で dB に線形）→ ピークが最も近い音を選び、残りの差を音量で合わせる（持ち上げは +12 dB まで）。シンバル以外は、同じ種類の次の打の立ち上がりから 5 ms で前の音を止める（チョーク）。キットの差し替えはロックの試行で受け渡し、音を鳴らすスレッドではメモリ確保・ロック待ち・ファイル入出力をしない |
| `Source/BuiltInSounds.*` | コード合成の仮の音（`kit.json` が無いときの既定のキット） |
| `Source/PluginProcessor.*`、`PluginEditor.*` | パラメータ（Gain）、状態の保存（キットのパスを含む）、画面（「kit.json を読む」「仮の音に戻す」、キットの内訳の表示） |
| `Tools/KitRenderTest.cpp` | DAW 無しでエンジンを通す確認用のコンソールプログラム（`kit.json` → 8 ビートの MIDI → WAV）。`--events 打点.csv`（1 行に「サンプル位置,ノート番号,強さ」）で任意の打点を鳴らし、音量をそろえずに 32 bit float で書く。CI では Python の `make_demo_kit.py` が作ったキットで毎回通し、**同じ打点を Python の `render_midi` で鳴らした音と 1 サンプル単位で比べる**（`analysis/engine_parity.py`。差の最大 5e-7） |

## ビルド（Windows、Visual Studio 2022）

```
cmake -S plugin -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target DrumSamplerPlugin_VST3 DrumSamplerPlugin_Standalone KitRenderTest
```

出力: `build/DrumSamplerPlugin_artefacts/Release/VST3/DrumSamplerPlugin.vst3`、`.../Standalone/DrumSamplerPlugin.exe`、`build/KitRenderTest_artefacts/Release/KitRenderTest.exe`

ビルドせずに試すなら、GitHub Actions の `build` の成果物 `DrumSamplerPlugin-windows-standalone`（exe）と `DrumSamplerPlugin-windows-vst3` を使う。

JUCE は CMake の FetchContent で `8.0.15` を取る。手元の JUCE を使うなら `-DJUCE_SOURCE_DIR=<JUCE のフォルダ>`。

## キットを鳴らす

1. 実ステムからキットと MIDI を作る（解析の参照実装。代表音は種類ごとに 24 音）。

   ```
   python analysis\analyze_stem.py data\stems\a04\drum.wav data\kits\a04 --model analysis\models\crnn_egmd_v5.onnx --start 71 --seconds 60 --preview
   ```

   `data\kits\a04\kit.json`、`kick\`、`snare\`… の WAV、`hits.mid`（打点。時刻は元のステムの先頭から、120 BPM）ができる。`--preview` を付けると、
   このキットを `hits.mid` だけで鳴らした音（`preview_midi.wav`。プラグインで鳴る音の予想）と、同じ区間の元の音（`original_cut.wav`）も書く。
   `--start` は解析を始める位置（秒。`run_eval.py` の表の「開始秒」か、曲フォルダの `start.txt` の値。a04 は 71）、`--seconds` は解析する長さ（省略すると最後まで）。合成音のデモなら `python analysis\make_demo_kit.py data\kits\demo`。

2. DAW でプラグイン（VST3）を立ち上げ、「kit.json を読む」からその `kit.json` を選ぶ。画面に、種類ごとの音の数と、強さ 1〜127 が対応する音量の範囲が出る。
3. DAW のテンポを 120 BPM にして、`hits.mid` をプラグインのトラックの先頭（1 小節目の頭）に置く。元のステム（`drum.wav`）を別のトラックの先頭に置けば、
   切り替えて聞き比べられる。プラグインは 2 ms の遅れをホストに伝えるので、立ち上がりは元のステムと同じ位置に来る。
4. MIDI の 36（キック）、38（スネア）、42（ハイハット）、45（タム）、49（シンバル）で鳴る。強さ 127 がその曲のその種類の最も大きい打、1 がそこから 30 dB 下
   （それより小さい打がある曲では、その音量）。

DAW 無しで確かめるなら:

```
build\KitRenderTest_artefacts\Release\KitRenderTest.exe data\stems\a07\kit\kit.json out.wav 8
```

## pluginval

```
pluginval --strictness-level 5 --validate-in-process --validate build/DrumSamplerPlugin_artefacts/Release/VST3/DrumSamplerPlugin.vst3
```

GitHub Actions（`.github/workflows/build.yml`）で Windows は pluginval、Linux は `KitRenderTest` の通しと Python の参照実装との一致、macOS はコンパイルの確認。参照実装の側（`analysis/drumeval/extract.py` など）だけを変えたときは `engine-parity.yml` が同じ一致の検査をする。Linux では画面のテストを除いた pluginval（`--skip-gui-tests`）が SUCCESS（2026-11-03、鳴らし方を変えた後の 2026-11-16 も）。

## ライセンスの注意

- JUCE は AGPLv3 と商用ライセンスの二本立て。JUCE 8 の Starter（年 $20,000 まで）はスプラッシュ画面不要。販売前の条件は `docs/phase0/ライセンス一覧.md`。
- VST3 SDK は JUCE に同梱。JUCE 8.0.15 の同梱版は 3.8.0 で MIT。VST の名称・ロゴを使うなら Steinberg の商標ガイドラインに従う。

## 次の段階

解析（対数メル、CRNN の推論、NMFD、切り出し、代表音の選択）を C++／ONNX Runtime に移す（`docs/plan/Phase1作業順.md`）。
