"""採譜の一致の検査（transcribe_parity.py）そのものを確かめる。C++ の出力の代わりに、Python の結果を同じ形式で書いたものを使う。

本物の C++（`TranscribeTest`）との比較は CI の transcribe-parity で行う。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transcribe_parity  # noqa: E402


def _write_like_cpp(work: Path, name: str, feature_shift: float = 0.0, prob_shift: float = 0.0, drop_hit: bool = False, time_shift: float = 0.0) -> None:
    ref = np.load(work / f"python_{name}.npz")
    out = work / f"cpp_{name}"
    out.mkdir(exist_ok=True)
    feat = ref["features"].copy()
    feat[3, 5] += feature_shift
    prob = ref["probs"].copy()
    prob[7, 0] += prob_shift
    feat.astype(np.float32).tofile(out / "features.f32")
    prob.astype(np.float32).tofile(out / "probs.f32")
    rows = [(t + time_shift, c) for t, c in zip(ref["times"], ref["classes"])]
    if drop_hit:
        rows = rows[:-1]
    (out / "hits.csv").write_text("time_sec,class,probability\n" + "".join(f"{t:.9f},{c},0.9\n" for t, c in rows), encoding="utf-8")
    (out / "info.json").write_text(json.dumps({"seconds": 1.0, "frames": int(feat.shape[0]), "mels": int(feat.shape[1]), "milliseconds": {}}), encoding="utf-8")


def test_make_and_check(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    work = tmp_path / "work"
    assert transcribe_parity.main(["make", str(work), "--seconds", "6"]) == 0
    assert {f"{n}.wav" for n in transcribe_parity.STEMS} <= {p.name for p in work.iterdir()}
    ref = np.load(work / "python_stereo_48000.npz")
    assert ref["features"].shape[1] == 64 and ref["probs"].shape[0] == ref["features"].shape[0] and len(ref["classes"]) > 5

    def check(**changes) -> int:
        for name in transcribe_parity.STEMS:
            _write_like_cpp(work, name)
        _write_like_cpp(work, "mono_96000", **changes)
        code = transcribe_parity.main(["check", str(work)])
        capsys.readouterr()
        return code

    assert check() == 0
    assert check(feature_shift=1e-6, prob_shift=1e-5) == 0  # float の丸めの程度の差は通す
    assert check(feature_shift=1e-2) == 1  # 対数メルが違う
    assert check(prob_shift=1e-2) == 1  # 確率が違う
    assert check(drop_hit=True) == 1  # 打が 1 つ足りない
    assert check(time_shift=0.01) == 1  # 打点が 1 フレームずれている
    # C++ の出力が無い
    for p in (work / "cpp_mono_16000").iterdir():
        p.unlink()
    assert transcribe_parity.main(["check", str(work)]) == 1
    assert "C++ の出力が無い" in capsys.readouterr().out


def test_onnxruntime_is_loaded_with_telemetry_off(monkeypatch):
    """onnxruntime を読み込む前に、利用状況の送信を切る環境変数を置く。利用者が値を入れていれば、それを変えない。"""
    pytest.importorskip("onnxruntime")
    from drumeval.infer import import_onnxruntime

    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    assert import_onnxruntime().__name__ == "onnxruntime"
    import os

    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    import_onnxruntime()
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "0"
    # スクリプトは onnxruntime を直接 import しない（必ず import_onnxruntime を通す）
    root = Path(__file__).resolve().parents[1]
    for path in list(root.glob("*.py")) + list((root / "drumeval").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        direct = [line for line in text.splitlines() if line.strip().startswith(("import onnxruntime", "from onnxruntime"))]
        assert not direct or path.name == "infer.py", (path.name, direct)
