# E-GMD で採譜モデルを学習する手順（Windows、RTX 3060 6 GB）

E-GMD（Expanded Groove MIDI Dataset）は Google Magenta が CC BY 4.0 で公開する、電子ドラム（Roland TD-17、43 キット）の演奏 444 時間の音声と MIDI。音声と MIDI は 2 ms 以内に揃っている。出典: https://magenta.tensorflow.org/datasets/e-gmd 、Callender, Hawthorne, Engel (2020) arXiv:2004.00188。ライセンスの整理は `ライセンス一覧.md` #1。

## 1. ダウンロードと展開（初回だけ、数時間）

- 全体（音声 + MIDI）: https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip （**zip 90 GB、展開後 132 GB**。展開中は両方あるので 230 GB ほど空けておく。展開が終わったら zip を消してよい）
- 置き場の例: `D:\data\e-gmd-v1.0.0\`（中に `e-gmd-v1.0.0.csv` と `drummer1\...` がある状態）。
- 7-Zip なら一部のドラマーだけ展開することもできる（例: `drummer1`〜`drummer3` だけで数十 GB）。CSV は必ず展開する。学習スクリプトは、音声と MIDI の両方があるトラックだけを使う。

### D ドライブだけで行う（C に空きが無い場合）

ブラウザの既定の保存先は C の「ダウンロード」なので使わず、PowerShell の `curl.exe`（Windows 10 以降に入っている）で D に直接落とす。途中で切れても同じコマンドで続きから再開できる（`-C -`）。

```
New-Item -ItemType Directory -Force D:\data | Out-Null
cd D:\data
curl.exe -L -C - -o e-gmd-v1.0.0.zip https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip
```

- 90 GB なので回線によっては数時間。PowerShell の窓は閉じずに置いておく。切れたら同じ 3 行目をもう一度実行する。
- 終わったらサイズを確かめる: `(Get-Item D:\data\e-gmd-v1.0.0.zip).Length / 1GB`（約 90 と出ればよい）。

展開も D の中で行う（Windows 10 以降の `tar` は zip を展開できる。`Expand-Archive` は大きな zip に時間がかかり C の一時領域を使うことがあるので使わない）。

```
cd D:\data
tar -xf e-gmd-v1.0.0.zip
dir D:\data\e-gmd-v1.0.0
```

- `e-gmd-v1.0.0.csv` と `drummer1` … が見えれば成功。見えずに `D:\data\e-gmd-v1.0.0\e-gmd-v1.0.0\...` のように一段深くなっていたら、`--egmd` にはその深い方のフォルダを渡す（CSV のあるフォルダ）。
- 展開が終わって中身を確かめたら `Remove-Item D:\data\e-gmd-v1.0.0.zip` で 90 GB を空けてよい。
- 7-Zip が入っていれば一部だけの展開もできる: `& "C:\Program Files\7-Zip\7z.exe" x e-gmd-v1.0.0.zip -oD:\data "e-gmd-v1.0.0\e-gmd-v1.0.0.csv" "e-gmd-v1.0.0\drummer1" "e-gmd-v1.0.0\drummer3"`。

pip のキャッシュと一時フォルダも C に作られるので、CUDA 版 PyTorch（約 2.5 GB）を入れる前に D に向ける（その PowerShell の窓の中だけ有効）。

```
$env:PIP_CACHE_DIR = "D:\data\pip-cache"
$env:TEMP = "D:\data\tmp"; $env:TMP = "D:\data\tmp"
New-Item -ItemType Directory -Force D:\data\tmp | Out-Null
```

学習のチェックポイントと ONNX はリポジトリの `analysis\models\` に書かれる（リポジトリは D にあるので C は使わない）。

## 2. CUDA 版の PyTorch を入れる（初回だけ）

リポジトリのフォルダの PowerShell で（`.venv` を有効にした状態）:

```
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r analysis\requirements-ml.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最後の行が `True` と `NVIDIA GeForce RTX 3060 Laptop GPU` を出せばよい。`cu126` で入らなければ https://pytorch.org/get-started/locally/ で Windows / pip / CUDA 12.x の行を選んでそのコマンドを使う。NVIDIA のドライバーは最新にしておく。

## 3. 学習

```
python analysis\train_transcriber.py --egmd D:\data\e-gmd-v1.0.0 --bleed-dir data\stems --epochs 8 --out analysis\models\crnn_egmd_v1.onnx
```

- 既定: 学習トラック 4,000 本（乱数で選ぶ。キットが散る）、1 本から 8 秒の区間を 2 つ、バッチ 8、GRU 96、OneCycle、8 エポック。1 エポック 1,000 ステップ。
- `--bleed-dir data\stems` は、区分 A のドラム以外のステム（ギター・ベース・ピアノ・ボーカル・その他）を「分離の残り」として混ぜる増強。ドラム系の名前のファイルは自動で除く。**CD 曲のステムを混ぜたモデルは評価用に限る**（製品のモデルは権利の明確な音だけで作る。R3）。
- 毎エポック、validation 分割 40 本 × 30 秒で種類ごとの F 値を出し、しきい値 0.3／0.4／0.5 のうち M0-1 が最も高いものを記録する。最良のエポックの ONNX が `--out` に残る（チェックポイントは `.pt`）。
- 時間の目安: 特徴量の計算が CPU なので、1 エポック 5〜15 分程度（ワーカー 2）。`--workers 4` で速くなることがある。DataLoader で止まるときは `--workers 0`。
- 学習データを増やすなら `--max-tracks 0`（全部、1 エポックが 10 倍）、区間を増やすなら `--chunks-per-track 4`。
- メモリ: GRU 96 とバッチ 8 × 8 秒なら VRAM 1 GB 未満。`--batch 16 --hidden 128` でも 6 GB に収まる。

## 4. 評価

E-GMD の test 分割（正解は MIDI なので本当の F 値。ただし電子ドラムの音に対して）:

```
python analysis\eval_egmd.py analysis\models\crnn_egmd_v1.onnx D:\data\e-gmd-v1.0.0 --split test --max-tracks 60
```

区分 A の実ステム（疑似正解に対する目安）:

```
python analysis\run_eval.py data\stems --model analysis\models\crnn_egmd_v1.onnx --render --out data\results\YYYY-MM-DD_egmd_v1.md
```

出てきた数字（E-GMD test の M0-1、区分 A の M0-1）を貼ってもらえれば、`docs/phase0/結果/` に記録する。M0-1 の目安は 0.85。

## 5. うまくいかないとき

- `CUDA out of memory` → `--batch 4` か `--hidden 64`。
- 学習が始まらない／ワーカーで止まる → `--workers 0`。
- 検証の F が上がらない → まず `--no-augment` で増強を切って比べる。増強が強すぎる場合は `drumeval/egmd.py` の `augment` の確率を下げる。
- E-GMD の CSV が見つからない → `--egmd` に渡すフォルダに `e-gmd-v1.0.0.csv` があるか確かめる。
