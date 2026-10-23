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

- 正解・推定の打点は CSV（`time_sec,class[,velocity]`）か MIDI（GM ドラムマップ、チャンネル10）。種類の名前は `drumeval/labels.py` の `FINE_TO_COARSE` を見る。採点では粗い5種（kick / snare / hihat / tom / cymbal）にまとめる。
- 時刻の許容誤差は ±50 ms。M0-1 は kick / snare / hihat の F 値の単純平均。
- STFT 距離は窓長 2048〜128 の5段で、スペクトル収束と対数振幅 L1 の和の平均。小さいほど近い。

## テスト

```
cd analysis && python -m pytest -q tests
```

合成データで指標の動き（完全一致で 1、許容誤差の外で 0、EQ を変えると距離が増える、など）を確かめる。

## データの置き場（コミットしない）

```
data/
  stems/<曲ID>/drums.wav          分離済みのドラムのステム
  stems/<曲ID>/hits.csv | .mid    打点の正解（あれば）
  stems/<曲ID>/sections.csv       区間のラベル（start_sec,end_sec,label）
```

曲ID と出どころは `docs/phase0/評価用ステム一覧.md` に書く。
