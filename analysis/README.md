# analysis — Phase 0 の Python 試作と採点

計画書 3.7 の「解析は Python で試作し、精度を確かめてから C++／ONNX に移す」に対応する場所。
採譜モデルや切り出しの試作もここに足す。

## 準備

```
python3 -m venv .venv
. .venv/bin/activate          # Windows は .venv\Scripts\activate
pip install -r analysis/requirements.txt
# 採譜モデルの学習と ONNX（任意）
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r analysis/requirements-ml.txt
```

## 採点スクリプト

| スクリプト | 指標 | 使い方 |
|---|---|---|
| `score_transcription.py` | 採譜の F 値（M0-1） | `python analysis/score_transcription.py 正解.csv 推定.csv` |
| `stft_distance.py` | 多重解像度 STFT 距離（M0-3、切り出しの信頼度） | `python analysis/stft_distance.py 元のステム.wav 再現.wav` |
| `baseline_transcribe.py` | 学習モデル無しの採譜（基準値） | `python analysis/baseline_transcribe.py ステム.wav 推定.csv` |
| `extract_kit.py` | 重ならない打の切り出し → 強さ別の層 → キット（M0-2 の材料） | `python analysis/extract_kit.py ステム.wav 打点.csv 出力先 --render 再現.wav` |
| `fit_fx.py` | 固定の形のエフェクトチェーンの当てはめ（M0-3） | `python analysis/fit_fx.py 乾いた音.wav 目標.wav --out params.json --render 当てはめ後.wav` |
| `pack_stems.py` | 分離したステムを受け渡し用に小さくする（モノラル 44.1 kHz FLAC、先頭 90 秒） | `python analysis/pack_stems.py 分離したWAVのフォルダ data/stems --seconds 90` |
| `run_eval.py` | 評価用ステムの集まりに対して3指標をまとめて出す（回帰テスト）。パート別ステム（キック.wav など）があれば疑似正解で F 値の目安を出し、パート別の切り出しも試す。`--model` で ONNX の採譜モデル、`--render` で聞き比べ用 WAV | `python analysis/run_eval.py data/stems --out docs/phase0/結果_2026-11-01.md` |
| `list_itunes_library.py` | iTunes の書き出し（XML かテキスト）をジャンル別の一覧にする（区分 A の選曲用） | `python analysis/list_itunes_library.py ライブラリ.xml --out docs/phase0/itunes一覧.md` |
| `make_eval_midi.py` | 区分 B の評価用 MIDI 曲（正解の打点・区間ラベル付き）を生成 | `python analysis/make_eval_midi.py --out analysis/eval_midi` |
| `remap_midi_notes.py` | ドラムのノート番号をキットのパッド配置に付け替える（GM → Maschine／XO など） | `python analysis/remap_midi_notes.py song.mid out.mid --preset maschine16` |
| `render_midi_synth.py` | song.mid を合成音で WAV にする（DAW で書き出す前の確認用） | `python analysis/render_midi_synth.py analysis/eval_midi/b01_rock/song.mid data/preview/b01_rock.wav` |
| `train_transcriber.py` | 合成データで小さな CRNN を学習し ONNX に書き出す | `python analysis/train_transcriber.py --steps 600` |
| `transcribe_onnx.py` | ONNX の採譜モデルを ONNX Runtime で動かす | `python analysis/transcribe_onnx.py analysis/models/crnn_synth.onnx ステム.wav 推定.csv` |

- 正解・推定の打点は CSV（`time_sec,class[,velocity]`）か MIDI（GM ドラムマップ、チャンネル10）。種類の名前は `drumeval/labels.py` の `FINE_TO_COARSE` を見る。採点では粗い5種（kick / snare / hihat / tom / cymbal）にまとめる。
- 時刻の許容誤差は ±50 ms。M0-1 は kick / snare / hihat の F 値の単純平均。
- STFT 距離は窓長 2048〜128 の5段で、スペクトル収束と対数振幅 L1 の和の平均。小さいほど近い。

## 基準値の採譜と合成音（`drumeval/baseline.py`、`drumeval/synth.py`）

- `baseline.py`: スペクトル・フラックスで打点を拾い、打点直後の低域／中域のエネルギー比でキックとスネアを分け、高域のフラックスでハイハットを拾う。公開モデルや自前の CRNN（3〜4週目）が「これより良い」ことを確かめる下限。きれいな合成音では F=1.0、にじみ（`bleed=0.3`）を足すとキック・スネアは 0.6 程度に落ちる。スネアの打では高域も立ち上がるので、スネアと同時に「ハイハット」も出してしまう（切り出しではその打が「重なり」扱いになる）。この弱点は学習モデルで直す。
- `synth.py`: 打点ラベルから合成のドラム音を描画する。採点の通しテストと、「自社の合成データで学習する」道の最小版。`bleed`（低域の雑音）と `reverb_tail`（残響）で難しさを足せる。

## 切り出しとキット（`drumeval/extract.py`）

- 前 0.1 秒・後 0.15 秒に他の打が無い打だけを切り出し、次の打の直前（上限 1.5 秒）まで取る。
- 種類ごとにピークの大きさで層（既定 3）に分け、層の中の複数の打をラウンドロビン（既定 3）にする。打が少ない種類は層を減らす。
- 打点に強さが無ければ、打点直後 20 ms のピークを種類ごとの分布で 1〜127 に割り当てる（計画書 3.2）。
- `--render` で同じ打点をキットで鳴らし直し、元のステムとの STFT 距離を出す。聞き比べ（M0-2）はこの2つの WAV で行う。
- 尾の延長（次の打で切れた分）は Phase 2 の課題で、ここではやらない。

## エフェクトの当てはめ（`drumeval/fx.py`）

- チェーンは固定: 入力ゲイン → EQ（ローシェルフ 100 Hz、ピーキング、ハイシェルフ 5 kHz）→ リバーブ（RT60、高域の減衰、混ぜる量）。コンプ・歪み・ゲートはまだ無い。
- 初期値は解析的に出す（帯域ごとの長期スペクトルの差 → EQ とゲイン、目標の尾の減衰 → RT60）。そこから scipy の Powell 法（勾配なし）で多重解像度 STFT 距離を小さくする。
- 合成テスト（既知のパラメータを掛けた音を目標にする）では、距離が 4.8 → 0.2 まで下がり、パラメータも正解に近づく（`tests/test_fx.py`）。実ステムでは「乾いた音」の基準が要るので、`extract_kit.py --render` の再現や内蔵ライブラリの音を乾いた音として使う。
- PyTorch の微分可能な版（計画書 3.5 の手順 3）は、実ステムでこの形が効くと分かってから書く。

## 採譜モデル（`drumeval/features.py`、`model.py`、`infer.py`）

- 特徴量は対数メル（22.05 kHz、窓 1024、ホップ 220 ≒ 10 ms、64 バンド）。numpy だけで書き、C++ に同じ計算を移せるようにしている。
- モデルは小さな CRNN（畳み込み2段 → 双方向 GRU → 各フレーム・各種類の確率）。打点はしきい値と局所最大で拾う。
- `train_transcriber.py` は `synth.py` の乱数パターン（テンポ・強さ・音色の揺らぎ・にじみ・残響）で学習し、ONNX に書き出して ONNX Runtime と PyTorch の出力が一致することを確かめる。これは**道具立ての確認**で、実曲の精度の主張ではない。実データ（E-GMD など、CC BY 4.0）で学習するときは `make_batch` を差し替える。
- 学習したモデル（`models/*.onnx`）は原則コミットしない。例外として、合成データで学習した小さな `crnn_synth_v0.onnx`（657 KB）は手元での比較用にコミットしてある。

## テスト

```
cd analysis && python -m pytest -q tests
```

合成データで指標の動き（完全一致で 1、許容誤差の外で 0、EQ を変えると距離が増える、など）と、ラベル → 合成音 → 基準値の採譜 → 採点 の通し、切り出し → キット → 再生、エフェクトの当てはめ、学習 → ONNX → ONNX Runtime を確かめる。torch が無ければ学習のテストは飛ばす。

## 評価用の MIDI 曲（`eval_midi/`）

`make_eval_midi.py` で生成した 7 曲（ドラム ch 10 ＋ベース＋コード＋リード、`hits.csv` と `sections.csv` 付き）。自作の内容なのでコミットしてある。`data/stems/` にコピーし、DAW の VST で書き出して SpectraLayers で分離した `drums.wav` を足すと、正解付きの評価用ステムになる（手順は各フォルダの README）。

## データの置き場（コミットしない）

```
data/
  stems/<曲ID>/drums.wav | .flac  分離済みのドラムのステム
  stems/<曲ID>/hits.csv | .mid    打点の正解（あれば）
  stems/<曲ID>/sections.csv       区間のラベル（start_sec,end_sec,label）
```

曲ID と出どころは `docs/phase0/評価用ステム一覧.md` に書く。
