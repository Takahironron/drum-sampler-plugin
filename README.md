# drum-sampler-plugin

ドラムのステムからキットを自動で組み、エフェクト込みで再現し、基準パターンから曲の展開（Intro／Verse／Chorus など）を生成するドラム音源プラグイン（JUCE / VST3）。

現在は Phase 0（技術検証）です。計画書は [`docs/plan/ドラム音源プラグイン開発計画.md`](docs/plan/ドラム音源プラグイン開発計画.md)、Phase 0 の作業順と進み具合は [`docs/plan/Phase0作業順.md`](docs/plan/Phase0作業順.md)、Claude Code 向けの説明は [`CLAUDE.md`](CLAUDE.md) にあります。

## 構成

```
CLAUDE.md                     Claude Code が最初に読む説明（仕様・計画の要点・作業ルール）
docs/plan/                    開発計画、Phase 0 の作業順
docs/phase0/                  Phase 0 の記録（ライセンス一覧、評価用ステム一覧、M0 の判定）
docs/research/                既存製品の比較 Excel
analysis/                     Python の試作と採点スクリプト（F 値、STFT 距離）
plugin/                       JUCE の空のプラグイン（CMake、VST3／Standalone）
.github/workflows/            Windows ビルドと pluginval、採点スクリプトのテスト
scripts/build_feature_matrix.py  機能比較 Excel の生成スクリプト
data/                         評価用ステム（コミットしない）
```

## ローカルの Claude Code で作業を始める

1. Claude Code を入れる（未導入の場合）。

   ```
   npm install -g @anthropic-ai/claude-code
   ```

2. リポジトリをクローンして、そのフォルダで `claude` を起動する。

   ```
   git clone https://github.com/Takahironron/drum-sampler-plugin.git
   cd drum-sampler-plugin
   claude
   ```

   `CLAUDE.md` は起動時に自動で読み込まれます。

3. 最初に次のように頼む。

   ```
   CLAUDE.md と docs/plan の計画書を読んで、Phase 0 の最初の2週間の作業（9章）から始めて。
   まず作業の順番を提案して。
   ```

## Phase 0 で必要になる道具（目安）

- Python 3.10 以上（PyTorch、librosa、openpyxl など）
- CMake と Visual Studio 2022（C++ のデスクトップ開発）、JUCE
- pluginval
- 評価用のドラムのステム（リポジトリの外、または `data/` に置く。`data/` はコミットされません）
