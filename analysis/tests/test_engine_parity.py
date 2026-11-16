"""エンジンの一致の検査（engine_parity.py）そのものを確かめる。C++ の出力の代わりに、Python の音を遅らせたものを使う。

本物の C++ との比較は CI の build と engine-parity（`KitRenderTest --events`）で行う。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine_parity  # noqa: E402
import make_demo_kit  # noqa: E402
from drumeval.extract import SAMPLE_LEAD_SEC, load_kit  # noqa: E402
from drumeval.labels import COARSE_TO_NOTE  # noqa: E402


def test_make_and_check(tmp_path, capsys):
    import soundfile as sf

    kit_dir, work = tmp_path / "kit", tmp_path / "work"
    assert make_demo_kit.main([str(kit_dir), "--sr", "22050"]) == 0
    kit = load_kit(kit_dir)
    assert set(kit.level_range) == set(kit.parts)  # デモのキットにも、強さと音量の対応が入る
    assert engine_parity.main(["make", str(kit_dir), str(work), "--seconds", "4"]) == 0

    events = [tuple(int(v) for v in line.split(",")) for line in (work / "events.csv").read_text().splitlines() if not line.startswith("#")]
    assert events == sorted(events, key=lambda e: e[0])  # 位置の順（同じ位置の打は書いた順のまま）
    assert {note for _, note, _ in events} == {COARSE_TO_NOTE[c] for c in kit.parts}
    assert {1, 127} <= {vel for _, _, vel in events} and all(1 <= vel <= 127 for _, _, vel in events)
    assert any(a[0] == b[0] and a[1] == b[1] for a, b in zip(events, events[1:]))  # 同じ位置・同じ種類の重なりも試す

    py, sr = sf.read(work / "python.wav", dtype="float64")
    lead = int(round(SAMPLE_LEAD_SEC * sr))
    assert np.max(np.abs(py)) > 0.1

    def check(cpp: np.ndarray) -> int:
        sf.write(work / "cpp.wav", cpp.astype(np.float32), sr, subtype="FLOAT")
        code = engine_parity.main(["check", str(kit_dir), str(work)])
        capsys.readouterr()
        return code

    delayed = np.concatenate([np.zeros(lead), py])  # エンジンは音を頭から鳴らすので、立ち上がりは lead だけ後に来る
    assert check(delayed) == 0
    assert check(delayed + 1e-6 * np.random.default_rng(0).standard_normal(delayed.size)) == 0  # float の丸めの程度の差は通す
    assert check(py) == 1  # 遅れが無い（立ち上がりの位置が違う）
    assert check(np.concatenate([np.zeros(lead + 1), py])) == 1  # 1 サンプルのずれ
    assert check(delayed * 1.01) == 1  # 音量が 0.1 dB 違う
    assert check(np.zeros_like(delayed)) == 1  # 無音
