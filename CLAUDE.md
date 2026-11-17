# CLAUDE.md

このリポジトリは、Takahiro が個人で開発するドラム音源プラグインのものです。Phase 0（技術検証）は M0 判定「続ける」で終わり（`docs/phase0/M0判定.md`、2026-11-03 確定）、現在は Phase 1（MVP）です。作業順と進み具合は `docs/plan/Phase1作業順.md` にあります（Phase 0 の記録は `docs/plan/Phase0作業順.md`）。作業を終えたらその「進み具合」の表を更新する。

## やり取りのルール

- Takahiro とのやり取り、コミットメッセージ、ドキュメントは日本語で書く。
- 計画書の「要確認」は確かめていない項目。確かめたら計画書を更新し、出典を書く。
- 期間の見込みは根拠のある数字ではない。ライセンスの記述は種類の整理で、法的な判断ではない。

## プラグインの仕様（4点）

1. 曲から抽出したドラムのステムを読み込み、各パートのキットを画面に自動で割り当てる。
2. 各パートのキットを、エフェクト込みで再現する。
3. 既存製品の全機能を網羅しつつ、直感的な UI にする。
4. 基準パターンから Verse／Chorus／Intro などの曲の展開を複数生成する。

## 開発の条件

- JUCE（C++、CMake）の VST3。Windows を先に出し、macOS／AU（と CLAP）は Phase 5。
- 個人開発（週20〜30時間を仮定）、実時間で動くこと。
- JUCE は AGPLv3 と商用ライセンス（EULA）の二本立て。JUCE 8 の無料階層（Starter、年 $20,000 まで）はスプラッシュ画面が不要。売上が超えたら Indie。VST3 SDK は 3.8 以降 MIT（JUCE 8.0.15 の同梱版は 3.8.0）。詳細と出典は `docs/phase0/ライセンス一覧.md`。

## 計画の要点（詳細は `docs/plan/ドラム音源プラグイン開発計画.md`）

### 構成

- **解析はすべてオフライン（裏のスレッド）**、音を鳴らす実時間処理と切り離す。音を鳴らすスレッドでは推論しない。
- 解析の5段: ① 前処理（テンポ・拍。ホストのテンポを優先）→ ② 採譜（打点の時刻・種類・強さ → MIDI）→ ③ パート分け → ④ 一打の切り出し（強さ別の層、ラウンドロビン）→ ⑤ キット候補 → ⑥ エフェクトの当てはめ。
- キット候補は3通り: (a) ステムから切り出した音（初期値）、(b) 内蔵ライブラリの近い音（埋め込みで検索）、(c) モーダル合成のパラメータの当てはめ（Phase 2）。
- エフェクトは固定の形のチェーン（パートごと: EQ → コンプ → 歪み → ゲート。共通: リバーブ、センド・定位、バスコンプ）を合成による分析（analysis-by-synthesis）で当てはめる。目標は「正解の設定を当てる」ではなく「聞いて同じに聞こえ、後から直せる」。
- 採譜は CRNN 系などを PyTorch で学習し、ONNX にして ONNX Runtime（MIT）で動かす。

### 音源方式（ハイブリッド）

- サンプル再生を全パートの土台にする。
- キック・スネア・タムはモーダル合成（線形＋張力の変調＋ビーターの衝突＋空気の結合・響き線の簡易版。D2 の選択肢1→2→3の順）。
- シンバルは当面サンプル。非線形の板（von Kármán）は v1.x 以降（D3 仮説2）。
- 電子系はアナログ風の合成（オシレーター＋ノイズ＋エンベロープ）。
- 各パートは「サンプル層」と「合成層」を比率で混ぜられる。

### モジュール

`Analyzer`（オフライン解析）、`KitModel`（キット・層・FX 設定のデータ）、`Engine`（`SampleVoice`・`ModalVoice`・`SynthVoice`、チョーク、FX チェーン、バス、個別出力）、`Sequencer` と `Arranger`（基準パターン、区間、展開の生成、ホスト同期）、`UI`。

### 開発の順序と品質

- 解析は Python（PyTorch、librosa など）で試作し、精度を確かめてから C++／ONNX に移す。音源側は最初から JUCE で書く。
- pluginval で VST3 を自動検査、GitHub Actions で Windows と macOS のビルド、解析の精度は固定の評価用ステム集で毎回測る（回帰テスト）。回帰テストは CI の analysis-regression（StemGMD 12 曲と E-GMD 14 本。基準は `analysis/regression_baseline.json`、比べるのは `analysis/check_regression.py`）。解析を直して数字が良くなったら、`--update` で基準を更新してコミットする。
- Takahiro の評価用ステム（区分 A、CD 曲）はクラウドに無い。パート分け・切り出し・再現を変えるときは、先にクラウドで E-GMD と StemGMD の標本（下の表）で前後を測り、それから Takahiro に区分 A での実行と聞き比べを頼む。
- **プラグインで鳴る音は、MIDI（時刻・種類・強さ）だけでキットを鳴らした音。** 物差しは `run_eval.py` の「MIDI だけで鳴らす」（`rerender_midi.wav`。キットは `extract.build_kit_by_level`）。方式 C（`rerender_parts.wav`）は MIDI に無い情報を使う（打ごとの音色と、位置の合わせ直し。距離は代表音の選びの乱数で、12 曲平均が ±0.03 揺れる）。キックやスネアだけの違いは、混ざった音の距離ではなく表の「種類ごとの距離」で見る。鳴らし方の決まり（キットと MIDI の約束）は `analysis/README.md`。Python の `render_midi` と C++ の `SampleEngine` は 1 サンプル単位で一致させてあり（CI の engine-parity）、変えるときは両方を変える。採譜の入口（間引き・対数メル・打点を拾う）も Python（`drumeval/features.py`、`infer.py`）と C++（`plugin/Source/Analysis/`）で一致させてある（CI の transcribe-parity）。
- **E-GMD の配布物は、正解の MIDI と音がファイルごとにずれているものがある**（test 分割の全部 5,271 本で、31% が 10 ms 以上、9% が 50 ms 以上）。学習と採点の前に `analysis/egmd_offsets.py` で測って直す（`label_offsets.csv`、評価用フォルダは `label_offset.txt`）。v5 までのモデルは直さないラベルで学習した。v5 の M0-1（E-GMD test の 60 本）は、直さない正解で 0.773、直した正解で 0.819（2026-11-17）。これより前の記録の E-GMD の数字は、直さない正解のもの。`train_transcriber.py` は、使うトラックにずれを測っていないものがあると学習を始めない。

### 機能の網羅（計画書4章）

- 「網羅」は「既存製品の各機能の**目的**を満たす」と読み替える。32機能を A（仕様の中核、15）／B（v1.0 まで、10）／C（v1.x 以降、6）／D（やらない・別の方法、1）に分け、v1.0 で 25/32 が目標。
- 網羅の度合いは `docs/research/ドラム音源_機能比較.xlsx` に自社の列を足して測る（`scripts/build_feature_matrix.py` に追記）。

### UI の原則（計画書5章）

- 作業の順に4画面（① 解析 ② キット ③ ミックス ④ 展開）。詳細欄は右側の1か所だけ、浮き窓や入れ子のダイアログを作らない。
- 基本表示のつまみは6個まで、「詳しく」で全パラメータ。
- 全画面の上部に「元のステム／再現」の A/B ボタンを常に置く。自動の結果は「提案」として出し、元に戻せる。

### 展開の生成（計画書6章）

- まずルールの組み合わせ（変換レシピ: 密度、楽器の置き換え、要素を抜く、強さ、ハーフ／ダブルタイム、区間の頭と終わりのフィル、揺れ）で作る（Phase 3）。学習モデルは Phase 5。

## フェーズ

| フェーズ | 目安 | 内容 |
|---|---|---|
| Phase 0 技術検証 | 1.5〜2か月 | Python で採譜・切り出し・エフェクトの当てはめを評価用ステム20本で試す。ライセンス確認。JUCE の空のプラグインでビルドと pluginval |
| Phase 1 MVP | 3〜4か月 | ステムを落とすとキットが鳴り、MIDI で DAW に渡せる |
| Phase 2 再現の質 | 3〜4か月 | エフェクトの当てはめ、内蔵ライブラリ検索、モーダル合成、細かい分類、信頼度表示 |
| Phase 3 曲の展開 | 2.5〜3か月 | パターン編集、変換レシピ、区間の並び、フィル（ベータ） |
| Phase 4 v1.0 | 2.5〜3か月 | 機能 B、UI 試験、インストーラー、Windows VST3 発売 |
| Phase 5 拡張 | 継続 | macOS／AU／CLAP、展開の学習モデル、物理パラメータの作り替え、非線形の板 |

### Phase 0 の判定（M0: 続けるかの判断）

- キック・スネア・ハイハットの採譜の F 値が 0.85 以上（目安）。
- 切り出した音で組んだキットを元のステムと聞き比べて、半数以上の曲で「同じキットに聞こえる」。
- エフェクトの当てはめで、多重解像度の STFT 距離が当てはめ前より明確に下がる。

判定が悪いときの変更順は計画書 7章の末尾を見る。

### Phase 0 の最初の2週間（計画書9章）

1. 評価用のドラムのステムを20本集める（自分の曲、権利の明確な素材、分離の道具を変えたもの）。区間（Intro／Verse／Chorus）も手でラベルを付ける。
2. 採譜の公開モデル・データのライセンス一覧を作る（E-GMD、Groove MIDI Dataset、ADTOF、StemGMD／LarsNet、Demucs）。
3. Python で「採譜 → 重ならない打の切り出し → 強さ別の層」を作り、5本で聞き比べる。
4. JUCE の空のプラグイン（サンプルを MIDI で鳴らすだけ）を CMake で作り、Windows でビルドして pluginval を通す。

## 作業ルール

- **著作権（R3）**: 市販曲から切り出した音や、市販曲のステムをリポジトリにコミットしない。評価用ステムはリポジトリの外（例: `data/`、`.gitignore` 済み）に置く。
- **OSS**: STK（MIT 形式）、Mutable Instruments のファームウェア（MIT）、ONNX Runtime（MIT）、Demucs（MIT）は組み込める。RipplerX・Kit・nlm などの GPL 系はコードをコピーせず、読んで仕組みを学んで自分で書き直す。
- **学習データ・モデル**: 製品に入れる採譜・パート分けモデルは、CC BY 4.0 のデータ（E-GMD、Groove MIDI Dataset、StemGMD）と自社の合成データから自分で学習する。ADTOF、LarsNet の学習済みモデル、Demucs の重み、madmom の同梱モデルは非商用なので評価・比較にだけ使う（`docs/phase0/ライセンス一覧.md`）。
- **増強に混ぜる音（にじみ）も権利の明確な音だけ**: 製品用のモデル（v6 から）は `analysis/make_bleed_set.py` の合成音を使う。CD 曲のステムを増強や微調整に使ったモデル（v1〜v5、`crnn_ft_*`）は評価用に限る。
- **特許（R9）**: 録音から材質を推定する米国特許（UNC、US 9,401,684 B2）がある。Phase 2 で推定を入れる前に請求項と権利の状態を確認する。
- **実時間処理**: オーディオスレッドでメモリの確保、ロック、ファイル入出力、推論をしない。
- **ONNX Runtime のテレメトリを切る**: 公式の配布物（C++ 版も pip 版も、1.31.0 で確認）は、利用状況を Microsoft に送る機能が既定で有効。Python では `onnxruntime` を直接 import せず `drumeval.infer.import_onnxruntime` を使う。C++ は `Transcriber` が初期化の前に切る（`docs/phase0/ライセンス一覧.md` の 6）。
- 範囲が膨らんだら、追加の要望は機能 C に入れて次の版で判断する（R6）。

## 資料の場所

| 資料 | 場所 |
|---|---|
| 開発計画（初版、2026-10-22） | `docs/plan/ドラム音源プラグイン開発計画.md` |
| Phase 1 の作業順と進み具合 | `docs/plan/Phase1作業順.md` |
| Phase 0 の作業順と進み具合（終了） | `docs/plan/Phase0作業順.md` |
| M0 判定（Phase 0 の結論） | `docs/phase0/M0判定.md` |
| ライセンス一覧（Phase 0） | `docs/phase0/ライセンス一覧.md` |
| 評価用ステム一覧（Phase 0） | `docs/phase0/評価用ステム一覧.md` |
| 手元（Windows）で評価を走らせる手順 | `docs/phase0/手元での評価の手順.md` |
| E-GMD で採譜モデルを学習する手順 | `docs/phase0/E-GMD学習の手順.md` |
| 評価の結果の記録 | `docs/phase0/結果/`（手元の出力は `data/results/`、コミットしない） |
| 採点スクリプト（F 値、STFT 距離） | `analysis/`（`analysis/README.md`） |
| クラウドでも使える実データの評価セット（CC BY 4.0。`data/` に取ってくる。コミットしない） | `analysis/fetch_egmd_sample.py`（E-GMD の 14 本。正解 MIDI、タム・シンバル・電子系のキット）、`analysis/fetch_stemgmd_sample.py`（StemGMD の 12 曲。パート別の単独録音 = 分離の本当の正解）。取った後 `python analysis/run_eval.py data/stemgmd_stems --parts nmfd-oracle`。E-GMD の標本は、正解と音の時刻のずれを `label_offset.txt` に書いてある（`run_eval.py` が直して使う） |
| 解析の参照実装（ステム 1 本 → キットと MIDI）と、キットと MIDI の約束 | `analysis/analyze_stem.py`、`analysis/drumeval/analyzer.py`、`analysis/README.md` |
| 解析の回帰テスト（基準と比べる） | `analysis/check_regression.py`、`analysis/regression_baseline.json`（CI: `analysis-regression.yml`） |
| Python と C++ のエンジンの一致の検査 | `analysis/engine_parity.py`（CI: `build.yml` の Linux、`engine-parity.yml`） |
| Python と C++ の採譜の一致の検査 | `analysis/transcribe_parity.py`、`plugin/Tools/TranscribeTest.cpp`（CI: `transcribe-parity.yml`、Windows と Linux） |
| JUCE のプラグイン（kit.json を読んで MIDI で鳴らす。解析の入口の C++ 版は `plugin/Source/Analysis/`） | `plugin/`（`plugin/README.md`） |
| 市販ドラム音源の機能比較（25製品×32機能、出典48件、ジャンル別キット） | `docs/research/ドラム音源_機能比較.xlsx` |
| 物理モデリング系プラグイン46件＋D3 先行事例の比較 | `docs/research/ドラム音源プラグイン比較.xlsx`（生成スクリプトは無い。元データは D1_products.md の表A〜C） |
| 機能比較 Excel の生成スクリプト（データはスクリプト内） | `scripts/build_feature_matrix.py`（`pip install openpyxl` の後 `python scripts/build_feature_matrix.py`） |

- 機能比較 Excel の空欄は「確認した出典に記載なし」で、機能が無いという意味ではない。

## 関連リポジトリ

- `Takahironron/music-genre-verification` の `research/drum-physical-modeling/`: ドラム物理モデリング研究。
  - `D1_products.md`: 製品・実装、OSS のライセンス、製品のパラメータ比較
  - `D2_methods.md`: 手法・文献、部位ごとのモデル化、JUCE で実時間に実装する選択肢
  - `D3_novelty.md`: 新規性・空白地帯、仮説1〜3、特許の確認状況
