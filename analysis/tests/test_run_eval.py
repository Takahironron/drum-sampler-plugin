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
        assert {f"rerender_parts_{tag}.wav", f"rerender_self_{tag}.wav", f"rerender_midi_{tag}.wav", f"nmfd_kick_{tag}.wav",
                f"kit_parts_audition_{tag}.wav"} <= names
    assert "original_cut.wav" in names and "rerender_parts.wav" not in names
    assert find_stem(song).name == "drum.wav"
    assert {c: p.name for c, p in find_part_stems(song).items()} == {"kick": "キック.wav", "snare": "スネア.wav", "hihat": "ハイハット.wav"}
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems"), "--tag", "bad tag"], capture_output=True, text=True)
    assert r.returncode != 0 and "--tag" in r.stderr


def test_product_path_writes_midi_only_render_and_matching_midi(tmp_path):
    """製品の経路（--parts nmfd）: モデルの打点から、キット（--save-kit）と MIDI（hits_model.mid）を書く。
    その 2 つだけで鳴らした音（rerender_midi.wav）の距離が表に出る。採譜の誤検出のうち音の無い打は MIDI に入らない。"""
    import pytest
    import soundfile as sf

    pytest.importorskip("onnxruntime")
    from drumeval import Hit, load_hits, multi_resolution_stft_distance
    from drumeval.extract import load_kit, render_midi
    from drumeval.labels import to_coarse

    sr = 22050
    truth = _sparse_pattern(8)
    dur = max(h.time for h in truth) + 1.2
    song = tmp_path / "stems" / "s01"
    song.mkdir(parents=True)
    # 録音と同じく、薄い雑音（−80 dB）を足す。合成の音は打の間が完全な無音で、そこでは距離の対数の項が 16／24 bit の丸めにも反応する
    x = render(truth, sr=sr, duration=dur) + 1e-4 * np.random.default_rng(0).standard_normal(int(dur * sr))
    sf.write(song / "drums.wav", x.astype(np.float32), sr)
    x, _ = sf.read(song / "drums.wav")
    save_hits_csv(truth, song / "hits.csv")
    out = tmp_path / "result.md"
    script = Path(__file__).resolve().parents[1] / "run_eval.py"
    model = Path(__file__).resolve().parents[1] / "models" / "crnn_synth_v0.onnx"
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "stems"), "--model", str(model), "--parts", "nmfd", "--render",
                        "--save-kit", "kit", "--start", "0", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    md = out.read_text(encoding="utf-8")
    assert "STFT（MIDI だけで鳴らす）" in md and "MIDI だけで鳴らしたときの距離の平均" in md and "F（MIDI にした打点" in md
    assert "MIDI だけの損失の内訳" in md and "位置が 5 ms 以上ずれる打の割合" in md
    assert {"rerender_midi.wav", "rerender_midi_km.wav", "hits_model.mid", "kit"} <= {p.name for p in song.iterdir()}
    assert "STFT（MIDI だけ、k-means のキット）" in md and "MIDI だけ・k-means のキット − 元" in md
    # 保存したキットと MIDI だけで、元に近い音が鳴る（プラグインに渡すもの）
    kit = load_kit(song / "kit")
    hits = [Hit(h.time, to_coarse(h.cls), h.velocity) for h in load_hits(song / "hits_model.mid")]
    assert set(kit.level_range) == set(kit.parts) and all(1 <= h.velocity <= 127 for h in hits)
    y = render_midi(hits, kit, duration=dur)
    y *= np.sqrt(np.mean(x**2)) / np.sqrt(np.mean(y**2))
    # 表の値（メモリ上で鳴らした音の距離）と同じ音になる。違いは MIDI の時刻の刻み（約 0.5 ms）とキットの WAV（24 bit）の分だけ
    lines = md.splitlines()
    heads = [h.strip() for h in lines[2].split("|")[1:-1]]
    reported = float([v.strip() for v in lines[4].split("|")[1:-1]][heads.index("STFT（MIDI だけで鳴らす）")])
    assert abs(multi_resolution_stft_distance(x, y)["total"] - reported) < 0.1
    row = dict(zip(heads, [v.strip() for v in lines[4].split("|")[1:-1]]))
    offsets = dict((t[:2], float(t[2:])) for t in row["位置が 5 ms 以上ずれる打の割合（MIDI だけ、種類別）"].split())
    assert offsets["ki"] == 0.0 and offsets["sn"] == 0.0  # 同じ音を並べた合成のステムでは、どの打も位置が合う
    # 種類ごとの距離（MIDI だけ／方式 C／自己再現）: 同じ音を並べたキックは、MIDI だけでも方式 C と変わらずそのパートの音に近い
    col = next(h for h in heads if h.startswith("種類ごとの距離"))
    part_d = {t[:2]: [float(v) for v in t[2:].split("/")] for t in row[col].split()}
    assert set(part_d) >= {"ki", "sn"} and all(len(v) == 3 for v in part_d.values())
    assert part_d["ki"][0] < part_d["ki"][1] + 0.1 and part_d["ki"][2] < 0.2, part_d
    assert "種類ごとの距離の平均（MIDI だけ／方式 C／自己再現）" in md
    assert reported < 0.7  # 合成音の v0 は、スネアの位置にハイハットを余分に出す（その分だけ遠い）
