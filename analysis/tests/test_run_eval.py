"""合成のステム集で run_eval が通ることを確かめる。"""

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import save_hits_csv  # noqa: E402
from drumeval.synth import render  # noqa: E402
from tests.test_extract import _sparse_pattern  # noqa: E402


def test_run_eval_on_synthetic_set(tmp_path):
    import soundfile as sf

    sr = 22050
    for i, bleed in enumerate((0.0, 0.2)):
        d = tmp_path / "stems" / f"s{i + 1:02d}"
        d.mkdir(parents=True)
        hits = _sparse_pattern(4)
        sf.write(d / "drums.wav", render(hits, sr=sr, bleed=bleed).astype(np.float32), sr)
        if i == 0:
            save_hits_csv(hits, d / "hits.csv")
    out = tmp_path / "result.md"
    script = Path(__file__).resolve().parents[1] / "run_eval.py"
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    md = out.read_text(encoding="utf-8")
    assert "| s01 |" in md and "| s02 |" in md
    assert "正解付きの曲: 1 / 2" in md
    assert "M0-1" in md


def test_pseudo_truth_from_part_stems(tmp_path):
    """パート別ステム（キック.wav など）だけがあるとき、疑似正解を作って F 値を出す。"""
    import soundfile as sf

    from drumeval import load_hits, transcription_scores
    from drumeval.labels import to_coarse
    from tests.test_pipeline import _rock_pattern

    sr = 22050
    ref = _rock_pattern(6)
    dur = max(h.time for h in ref) + 1.2
    song = tmp_path / "stems" / "a01"
    song.mkdir(parents=True)
    mix = np.zeros(int(dur * sr))
    for cls, name in (("kick", "キック"), ("snare", "スネア"), ("hihat", "ハイハット")):
        part = render([h for h in ref if to_coarse(h.cls) == cls], sr=sr, duration=dur)
        sf.write(song / f"{name}.wav", part.astype(np.float32), sr)
        mix += part
    sf.write(song / "drum.wav", (mix / np.max(np.abs(mix)) * 0.9).astype(np.float32), sr)
    sf.write(song / "ギター.wav", np.zeros(sr, dtype=np.float32), sr)  # 他のステムが並んでいても drum.wav を選ぶ

    script = Path(__file__).resolve().parents[1] / "run_eval.py"
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "疑似（パート別ステム）" in r.stdout
    pseudo = load_hits(song / "hits_pseudo.csv")
    s = transcription_scores(ref, pseudo)
    for c in ("kick", "snare", "hihat"):
        assert s[c].f_measure >= 0.95, (c, s[c])


def test_detect_drum_start_skips_silent_intro():
    from run_eval import detect_drum_start

    sr = 22050
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(sr * 8) * 1e-4, render(_sparse_pattern(6), sr=sr)])  # 8 秒の無音（微小雑音）の後にドラム
    start = detect_drum_start(x, sr)
    assert 7.0 <= start <= 9.0
    assert detect_drum_start(render(_sparse_pattern(4), sr=sr), sr) == 0.0


def test_render_tag_keeps_results_side_by_side(tmp_path):
    """--tag を付けると、書き出す WAV の名前の末尾に印が付く（モデルを替えた結果を上書きせずに並べて聞ける）。
    印の付いたファイルも、次の実行でステムやパート別ステムとして拾われない。"""
    import soundfile as sf

    from drumeval.labels import to_coarse
    from drumeval.stems import find_part_stems, find_stem
    from tests.test_pipeline import _rock_pattern

    sr = 22050
    ref = _rock_pattern(4)
    dur = max(h.time for h in ref) + 1.2
    song = tmp_path / "stems" / "a01"
    song.mkdir(parents=True)
    mix = np.zeros(int(dur * sr))
    for cls, name in (("kick", "キック"), ("snare", "スネア"), ("hihat", "ハイハット")):
        part = render([h for h in ref if to_coarse(h.cls) == cls], sr=sr, duration=dur)
        sf.write(song / f"{name}.wav", part.astype(np.float32), sr)
        mix += part
    sf.write(song / "drum.wav", (mix / np.max(np.abs(mix)) * 0.9).astype(np.float32), sr)
    script = Path(__file__).resolve().parents[1] / "run_eval.py"
    for tag in ("v5", "ft"):
        r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems"), "--parts", "nmfd-oracle", "--render", "--tag", tag],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    names = {p.name for p in song.iterdir()}
    for tag in ("v5", "ft"):
        assert {f"rerender_parts_{tag}.wav", f"rerender_self_{tag}.wav", f"nmfd_kick_{tag}.wav", f"kit_parts_audition_{tag}.wav"} <= names
    assert "original_cut.wav" in names and "rerender_parts.wav" not in names
    assert find_stem(song).name == "drum.wav"
    assert {c: p.name for c, p in find_part_stems(song).items()} == {"kick": "キック.wav", "snare": "スネア.wav", "hihat": "ハイハット.wav"}
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems"), "--tag", "bad tag"], capture_output=True, text=True)
    assert r.returncode != 0 and "--tag" in r.stderr
