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
python analysis\train_transcriber.py --egmd D:\data\e-gmd-v1.0.0 --bleed-dir data\stems --out analysis\models\crnn_egmd_v5.onnx
```

- 既定（v5 から）: 学習トラック 8,000 本（乱数で選ぶ。キットが散る）、1 本から 8 秒の区間を 2 つ、バッチ 8、64 メル帯域・周波数方向の縮小 2 回、対数メルの時間差分のチャンネル、GRU 160 × 2 層、重みの移動平均（EMA 0.999。生と EMA の良い方を残す）、OneCycle、16 エポック、合成音の混ぜ込み 15%、検証 160 本、ワーカー 4。1 エポック 2,000 ステップ（約 25〜40 分、16 エポックで 7〜10 時間。夜に回す）。種類ごとのしきい値（0.3〜0.8）を検証で選んで ONNX の設定に保存する。
- 時間を短くするなら `--max-tracks 4000`（v2 と同じ量、約 4 時間）。
- これまで（E-GMD test の M0-1）: v1 0.70、v2 0.74、v3 0.715（96 メル・縮小 1 回は効かなかった）、v4 0.773（時間差分のチャンネル、種類別しきい値、8,000 本。検証は 0.846）。v5 は EMA・2 層・重み・検証 160 本（`結果/` を見る）。
- 診断: `python analysis\eval_egmd.py <model> D:\data\e-gmd-v1.0.0 --split test --max-tracks 60 --diagnose --sweep` で、取りこぼし（小さい打か、キック・スネアと同時の打か）と誤検出（他の種類との混同か）の内訳、しきい値 0.3〜0.8 の F が出る。`--min-velocity 40` で velocity 40 未満の打（ゴーストノート）を除いた F 値。
- エポックごとの検証の結果は `analysis\models\crnn_egmd_v5.log.txt` に残る（これを貼ればよい）。
- `--bleed-dir data\stems` は、区分 A のドラム以外のステム（ギター・ベース・ピアノ・ボーカル・その他）を「分離の残り」として混ぜる増強。ドラム系の名前のファイルは自動で除く。**CD 曲のステムを混ぜたモデルは評価用に限る**（製品のモデルは権利の明確な音だけで作る。R3）。
- 毎エポック、validation 分割 40 本 × 30 秒で種類ごとの F 値を出し、しきい値 0.3／0.4／0.5 のうち M0-1 が最も高いものを記録する。最良のエポックの ONNX が `--out` に残る（チェックポイントは `.pt`）。
- 時間の目安: 特徴量の計算が CPU なので、1 エポック 5〜15 分程度（ワーカー 2）。`--workers 4` で速くなることがある。DataLoader で止まるときは `--workers 0`。
- 学習データを増やすなら `--max-tracks 0`（全部、1 エポックが 10 倍）、区間を増やすなら `--chunks-per-track 4`。
- メモリ: GRU 96 とバッチ 8 × 8 秒なら VRAM 1 GB 未満。`--batch 16 --hidden 128` でも 6 GB に収まる。

## 4. 評価

E-GMD の test 分割（正解は MIDI なので本当の F 値。ただし電子ドラムの音に対して）:

```
python analysis\eval_egmd.py analysis\models\crnn_egmd_v5.onnx D:\data\e-gmd-v1.0.0 --split test --max-tracks 60 --by-kit --diagnose --sweep
```

`--by-kit` でキットごとの M0-1 も出る（電子系のキットが弱いかを見る）。

区分 A の実ステム（疑似正解に対する目安）:

```
python analysis\run_eval.py data\stems --model analysis\models\crnn_egmd_v5.onnx --render --out data\results\YYYY-MM-DD_egmd_v5.md
```

出てきた数字（E-GMD test の M0-1、区分 A の M0-1）を貼ってもらえれば、`docs/phase0/結果/` に記録する。M0-1 の目安は 0.85。

## 5. 実ステムへの微調整（疑似ラベル、leave-one-out）

E-GMD で学習したモデルは、CD 曲の分離済みステムでは取りこぼしが多い（E-GMD の検証 0.87 → 区分 A の疑似正解 0.56）。
区分 A の疑似正解（`run_eval.py` が各曲のフォルダに書く `hits_pseudo.csv`。分離ツールのパート別ステムの打点検出）で短く学習し直し、
実ステムに合わせる。正直に測るため、**曲ごとに「その曲以外の 9 曲で学習 → その曲で測る」**（leave-one-out）を繰り返す。

```
python analysis\run_eval.py data\stems --model analysis\models\crnn_egmd_v5.onnx --parts nmfd-oracle --out data\results\YYYY-MM-DD_nmfd_oracle.md
python analysis\train_transcriber.py --finetune-stems data\stems --init-from analysis\models\crnn_egmd_v5.pt --holdout all --egmd D:\data\e-gmd-v1.0.0 --out analysis\models\crnn_ft_v5.onnx
python analysis\run_eval.py data\stems --model "analysis\models\crnn_ft_v5_{id}.onnx" --parts nmfd --render --out data\results\YYYY-MM-DD_ft_nmfd.md
```

- 1 行目は、`hits_pseudo.csv` を最新の形式（先頭のコメント行に評価の開始位置・区間の長さ・ラベルのあるパート）で書き直すため。どの `run_eval.py` の実行でも書き直される（古い形式でも動くが、区間の決め方を `run_eval.py` と揃えるため）。
- 2 行目が微調整。曲ごとに `crnn_ft_v5_<曲ID>.onnx`（と `.json`、`.pt`）、前後の表 `crnn_ft_v5_loo.md`、ログ `crnn_ft_v5.log.txt` が `analysis\models\` にできる。**`crnn_ft_v5_loo.md` を貼ってもらえれば記録する。**
  - 既定: 6 エポック、学習率 1e-4、EMA なし、1 曲から 8 秒の区間を 40、E-GMD の train から 200 本（× 2 区間）を混ぜる（E-GMD を忘れないため。`--egmd` を省けば実ステムだけ）。E-GMD の validation 40 本 × 30 秒で忘れ具合も出す。
  - 「前」と「後」のしきい値は、どちらも学習した 9 曲で種類ごとに選ぶ（評価曲は見ない）。前の数字は `run_eval.py` の表（v5 の ONNX のしきい値）と少し違う。
  - パートの無い種類（タムのステムが無い曲のタム）は正解が無いので損失から外す。
  - 時間の目安: 1 曲あたり数分（GPU）、10 曲で 1 時間前後。`--holdout a01` のように 1 曲だけでも試せる。`--holdout` を省くと 10 曲全部で学習し、`crnn_ft_v5.onnx` 1 つと当てはまりの表 `crnn_ft_v5_fit.md` を書く（汎化の目安にはならない。M0-2 の聞き比べ用）。
  - 過剰適応（学習した曲に寄り過ぎて、初期値が良かった曲が悪くなる。2026-11-13 の a06）には `--mix-init 0.5`: 微調整の後に初期値の重みを半々で混ぜる（robust fine-tuning）。E-GMD の忘れが大きければ `--max-tracks 600` で混ぜ込みを増やす。
  - 表には「前（初期値のしきい値）」（run_eval.py の表と同じ条件）と、E-GMD の検証を初期値のしきい値で測った列も出る（しきい値の違いと忘れ具合を分けて見る）。
  - `--init-from` は `train_egmd` のチェックポイント。v5 までの `.pt` は最終エポックの重み（ONNX は最良エポック）。v6 からは最良エポックが `.best.pt` にも残るので、そちらを渡す。
- 3 行目は、曲ごとの微調整モデルで製品の経路（NMFD）を測る。推論の間引きは学習と同じ多相フィルタ（2026-11-13 から。それ以前の表とは数字が少し変わる）。`{id}` が曲 ID に置き換わる（PowerShell では引用符で囲む）。表の「推定」の列に `crnn_ft_v5_a01` のように出る。
- **CD 曲のステムで学習したモデルは評価用に限る**（製品のモデルは権利の明確な音だけで作る。R3）。区分 B（正解 MIDI 付き）が揃えば、同じ仕組みで本当の正解でも学習できる。

## 6. うまくいかないとき

- `CUDA out of memory` → `--batch 4` か `--hidden 64`。
- 学習が始まらない／ワーカーで止まる → `--workers 0`。
- 検証の F が上がらない → まず `--no-augment` で増強を切って比べる。増強が強すぎる場合は `drumeval/egmd.py` の `augment` の確率を下げる。
- E-GMD の CSV が見つからない → `--egmd` に渡すフォルダに `e-gmd-v1.0.0.csv` があるか確かめる。
