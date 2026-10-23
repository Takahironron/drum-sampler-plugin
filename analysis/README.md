# analysis — Phase 0 の Python 試作と採点

計画書 3.7 の「解析は Python で試作し、精度を確かめてから C++／ONNX に移す」に対応する場所。
採譜モデルや切り出しの試作もここに足す。

## 準備

```
python3 -m venv .venv
. .venv/bin/activate          # Windows は .venv\Scripts\activate
pip install -r analysis/requirements.txt
```

## 採点スクリプト

| スクリプト | 指標 | 使い方 |
|---|---|---|
| `score_transcription.py` | 採譜の F 値（M0-1） | `python analysis/score_transcription.py 正解.csv 推定.csv` |
| `stft_distance.py` | 多重解像度 STFT 距離（M0-3、切り出しの信頼度） | `python analysis/stft_distance.py 元のステム.wav 再現.wav` |
| `baseline_transcribe.py` | 学習モデル無しの採譜（基準値） | `python analysis/baseline_transcribe.py ステム.wav 推定.csv` |
| `extract_kit.py` | 重ならない打の切り出し → 強さ別の層 → キット（M0-2 の材料） | `python analysis/extract_kit.py ステム.wav 打点.csv 出力先 --render 再現.wav` |
| `fit_fx.py` | 固定の形のエフェクトチェーンの当てはめ（M0-3） | `python analysis/fit_fx.py 乾いた音.wav 目標.wav --out params.json --render 当てはめ後.wav` |

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

## テスト

```
cd analysis && python -m pytest -q tests
```

合成データで指標の動き（完全一致で 1、許容誤差の外で 0、EQ を変えると距離が増える、など）と、ラベル → 合成音 → 基準値の採譜 → 採点 の通しを確かめる。

## データの置き場（コミットしない）

```
data/
  stems/<曲ID>/drums.wav          分離済みのドラムのステム
  stems/<曲ID>/hits.csv | .mid    打点の正解（あれば）
  stems/<曲ID>/sections.csv       区間のラベル（start_sec,end_sec,label）
```

曲ID と出どころは `docs/phase0/評価用ステム一覧.md` に書く。
