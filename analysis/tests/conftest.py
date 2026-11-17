"""テスト全体の設定。"""

import os

# onnxruntime の公式の配布物は、利用状況を Microsoft に送る機能（テレメトリ）が既定で有効（1.31.0 の Privacy.md）。
# テストが onnxruntime を直接読み込む前に切る（スクリプトの側は `drumeval.infer.import_onnxruntime` が切る）。
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
