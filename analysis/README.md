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

- 正解・推定の打点は CSV（`time_sec,class[,velocity]`）か MIDI（GM ドラムマップ、チャンネル10）。種類の名前は `drumeval/labels.py` の `FINE_TO_COARSE` を見る。採点では粗い5種（kick / snare / hihat / tom / cymbal）にまとめる。
- 時刻の許容誤差は ±50 ms。M0-1 は kick / snare / hihat の F 値の単純平均。
- STFT 距離は窓長 2048〜128 の5段で、スペクトル収束と対数振幅 L1 の和の平均。小さいほど近い。

## 基準値の採譜と合成音（`drumeval/baseline.py`、`drumeval/synth.py`）

- `baseline.py`: スペクトル・フラックスで打点を拾い、打点直後の低域／中域のエネルギー比でキックとスネアを分け、高域のフラックスでハイハットを拾う。公開モデルや自前の CRNN（3〜4週目）が「これより良い」ことを確かめる下限。きれいな合成音では F=1.0、にじみ（`bleed=0.3`）を足すとキック・スネアは 0.6 程度に落ちる。
- `synth.py`: 打点ラベルから合成のドラム音を描画する。採点の通しテストと、「自社の合成データで学習する」道の最小版。`bleed`（低域の雑音）と `reverb_tail`（残響）で難しさを足せる。

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
