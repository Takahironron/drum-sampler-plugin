"""解析の参照実装（ステム 1 本 → キットと MIDI）を、合成のステムで確かめる。"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit, load_hits, multi_resolution_stft_distance, save_hits_csv  # noqa: E402
from drumeval.analyzer import analyze, preview, save_analysis  # noqa: E402
from drumeval.extract import level_for_velocity, load_kit, render_midi  # noqa: E402
from drumeval.labels import to_coarse  # noqa: E402
from drumeval.synth import render  # noqa: E402
from tests.test_extract import _sparse_pattern  # noqa: E402

SR = 22050
SCRIPT = Path(__file__).resolve().parents[1] / "analyze_stem.py"


def _stem(tail_sec: float = 1.2) -> tuple[np.ndarray, list[Hit]]:
    truth = _sparse_pattern(8)
    x = render(truth, sr=SR, duration=max(h.time for h in truth) + tail_sec)
    return x, truth


def test_analyze_gives_a_kit_and_hits_that_play_back_like_the_stem():
    """打点（時刻と種類だけ）を渡すと、キットと、強さの付いた打点が返る。それを MIDI として鳴らすと元に近い。"""
    x, truth = _stem()
    a = analyze(x, SR, hits=[Hit(h.time, h.cls, None) for h in truth])  # 細かい種類（hihat_closed）は粗い種類に直る
    assert set(a.kit.parts) == {"kick", "snare", "hihat"}
    assert len(a.hits) == len(truth) and a.n_dropped == 0
    assert all(h.cls in a.kit.parts and 1 <= h.velocity <= 127 for h in a.hits)
    assert multi_resolution_stft_distance(x, preview(a))["total"] < 0.4
    for c in a.kit.parts:
        true_v = np.array([h.velocity for h in truth if to_coarse(h.cls) == c], dtype=float)
        est = [h for h in a.hits if h.cls == c]
        assert max(h.velocity for h in est) == 127  # 最も大きい打が 127
        # 強さから戻した音量は、元の打の音量（合成では 強さ／127 に比例）と 0.5 dB 以内で合う
        est_db = np.array([level_for_velocity(h.velocity, a.kit.level_range[c]) for h in est])
        true_db = 20 * np.log10(true_v / true_v.max())
        assert np.max(np.abs((est_db - est_db.max()) - true_db)) < 0.5
    assert set(a.timings) == {"採譜", "パート分け", "切り出しとキット"}


def test_analyze_drops_hits_where_the_part_has_no_sound():
    """採譜の誤検出のうち、そのパートに音の無い時刻のものは、MIDI にもキットにも入れない。"""
    x, truth = _stem(tail_sec=3.0)
    last = max(h.time for h in truth)
    false = [Hit(last + 1.6, "kick", None), Hit(last + 2.2, "kick", None),  # 無音の区間
             Hit(0.1 + 3.25 * 0.6, "kick", None)]  # ハイハットだけが鳴っている位置
    clean = analyze(x, SR, hits=truth)
    a = analyze(x, SR, hits=truth + false)
    assert a.n_dropped == len(false)
    assert [(round(h.time, 3), h.cls) for h in a.hits] == [(round(h.time, 3), h.cls) for h in clean.hits]
    assert multi_resolution_stft_distance(x, preview(a))["total"] < 0.4


def test_analyze_needs_a_model_or_hits_and_survives_an_empty_result():
    import pytest

    x, _ = _stem()
    with pytest.raises(ValueError):
        analyze(x, SR)
    a = analyze(np.zeros(SR), SR, hits=[])  # 打が 1 つも無い（無音のステム）
    assert a.hits == [] and a.kit.parts == {} and preview(a).size == SR


def test_save_analysis_writes_kit_midi_and_summary(tmp_path):
    """保存したキットと MIDI だけから、同じ音が鳴る（プラグインに渡すもの）。"""
    x, truth = _stem()
    a = analyze(x, SR, hits=truth)
    meta = save_analysis(a, tmp_path / "out", offset_sec=2.0)
    assert {p.name for p in (tmp_path / "out").iterdir()} >= {"kit.json", "hits.mid", "analysis.json", "kick", "snare", "hihat"}
    assert json.loads((tmp_path / "out" / "analysis.json").read_text(encoding="utf-8")) == meta
    assert meta["hits"] == {"kick": 16, "snare": 8, "hihat": 16} and meta["offset_sec"] == 2.0 and meta["dropped_silent_hits"] == 0
    assert set(meta["level_range_db"]) == set(meta["kit"]) == {"kick", "snare", "hihat"}

    kit = load_kit(tmp_path / "out")
    hits = load_hits(tmp_path / "out" / "hits.mid")  # 時刻は元のファイルの先頭から（offset_sec を足してある）
    assert [(to_coarse(h.cls), h.velocity) for h in hits] == [(h.cls, h.velocity) for h in a.hits]
    assert np.max(np.abs(np.array([h.time for h in hits]) - 2.0 - np.array([h.time for h in a.hits]))) < 0.001  # MIDI の刻みは約 1 ms
    y = render_midi([Hit(h.time - 2.0, h.cls, h.velocity) for h in hits], kit, duration=a.duration)
    assert multi_resolution_stft_distance(x, y)["total"] < 0.4


def test_analyze_stem_cli_with_given_hits(tmp_path):
    import soundfile as sf

    x, truth = _stem()
    lead = np.zeros(int(1.5 * SR))  # 頭に 1.5 秒の無音。--start で飛ばす
    sf.write(tmp_path / "drums.wav", np.concatenate([lead, x]).astype(np.float32), SR)
    save_hits_csv([Hit(h.time + 1.5, h.cls, h.velocity) for h in truth], tmp_path / "hits.csv")
    out = tmp_path / "kit"
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "drums.wav"), str(out), "--hits", str(tmp_path / "hits.csv"),
                        "--start", "1.5", "--preview"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = re.search(r"MIDI だけで鳴らした音と元の距離（多重解像度 STFT）: ([0-9.]+)。", r.stdout)
    assert m and float(m.group(1)) < 0.4, r.stdout
    assert {p.name for p in out.iterdir()} >= {"kit.json", "hits.mid", "analysis.json", "preview_midi.wav", "original_cut.wav"}
    hits = load_hits(out / "hits.mid")
    assert len(hits) == len(truth)
    assert abs(hits[0].time - (truth[0].time + 1.5)) < 0.003  # MIDI の時刻は元のファイルの先頭から
    y, sr = sf.read(out / "preview_midi.wav")
    ref, _ = sf.read(out / "original_cut.wav")
    assert sr == SR and y.size == ref.size == x.size
    assert np.corrcoef(ref, y)[0, 1] > 0.9  # 聞き比べ用の 2 本は、同じ長さ・同じ位置・同じ音量（WAV は 16 bit なので、距離は標準出力の値で見る）
    assert abs(np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(ref**2)) - 1.0) < 0.02


def test_analyze_stem_cli_with_the_transcription_model(tmp_path):
    """採譜モデル（合成データで学習した v0）から通す。モデルも打点も無ければ使い方の誤り。"""
    import pytest
    import soundfile as sf

    pytest.importorskip("onnxruntime")
    x, truth = _stem()
    sf.write(tmp_path / "drums.wav", x.astype(np.float32), SR)
    model = Path(__file__).resolve().parents[1] / "models" / "crnn_synth_v0.onnx"
    out = tmp_path / "kit"
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "drums.wav"), str(out), "--model", str(model)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    meta = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert meta["model"] == "crnn_synth_v0.onnx"
    assert abs(meta["hits"]["kick"] - 16) <= 2 and abs(meta["hits"]["snare"] - 8) <= 2  # 合成の音は v0 でほぼ拾える
    assert not (out / "preview_midi.wav").exists()  # --preview を付けなければ書かない
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "drums.wav"), str(out)], capture_output=True, text=True)
    assert r.returncode != 0 and "--model" in r.stderr
