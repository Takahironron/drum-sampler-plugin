# b06_ballad: バラード（遅い、ライド）（72 BPM、133 秒、打点 559）

1. `song.mid` を DAW に読み込み、Drums（ch 10）に生ドラム系の VST、Bass／Chords／Lead に任意の VST を割り当てる。
2. 1 小節目の頭（0 秒）から 2mix を WAV で書き出す（先頭に無音を足さない）。
3. SpectraLayers（や Demucs）でドラムを分離し、`drums.wav` としてこのフォルダに置く。
4. `python analysis/run_eval.py data/stems` で F 値を出す。

`hits.csv` は `song.mid` のドラムの打点（正解）、`sections.csv` は区間のラベル。
