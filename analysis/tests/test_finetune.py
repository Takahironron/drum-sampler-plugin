"""実ステムの疑似ラベルでの微調整（leave-one-out）の通し。小さな合成の曲で、読み込み → 学習 → ONNX → まとめの表 を確かめる。torch が無ければ飛ばす。"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drumeval import Hit, load_hits, save_hits_csv  # noqa: E402
from drumeval.egmd import CLASSES5  # noqa: E402
from drumeval.features import FeatureConfig  # noqa: E402
from drumeval.labels import read_hits_csv_meta, to_coarse  # noqa: E402
from drumeval.stems import StemChunks, read_stem_songs, resolve_start  # noqa: E402
from drumeval.synth import render  # noqa: E402
from tests.test_pipeline import _rock_pattern  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _make_song(song: Path, ref: list[Hit], sr: int = 22050, lead_in: float = 0.0, parts=(("kick", "キック"), ("snare", "スネア"), ("hihat", "ハイハット"))) -> None:
    """パート別ステム（合成）と drum.wav。`lead_in` 秒の無音を頭に足す。"""
    import soundfile as sf

    song.mkdir(parents=True)
    dur = max(h.time for h in ref) + 1.0
    pad = np.zeros(int(lead_in * sr))
    mix = np.zeros(int(dur * sr) + pad.size)
    for cls, name in parts:
        part = np.concatenate([pad, render([h for h in ref if to_coarse(h.cls) == cls], sr=sr, duration=dur)])
        sf.write(song / f"{name}.wav", part.astype(np.float32), sr)
        mix += part
    sf.write(song / "drum.wav", (mix / (np.max(np.abs(mix)) + 1e-9) * 0.9).astype(np.float32), sr)


def _init_checkpoint(path: Path, cfg: FeatureConfig, hidden: int = 8, old_format: bool = False) -> None:
    """`old_format` は v5 までのチェックポイント（weights・thresholds の項目が無い）を模す。"""
    from drumeval.model import DrumCRNN

    model = DrumCRNN(cfg.n_mels, len(CLASSES5), hidden=hidden, pools=2, delta=True, layers=1)
    ck = {"state_dict": model.state_dict(), "ema_state_dict": model.state_dict(), "hidden": hidden, "layers": 1, "pools": 2, "delta": True,
          "classes": CLASSES5, "features": cfg.__dict__, "epoch": 1}
    if not old_format:
        ck.update({"weights": "raw", "thresholds": {c: 0.5 for c in CLASSES5}})
    torch.save(ck, path)


def test_read_stem_songs_and_chunks(tmp_path):
    stems = tmp_path / "stems"
    ref = _rock_pattern(6)
    _make_song(stems / "s01", ref, lead_in=3.0)
    # run_eval が疑似正解を書く（コメント行に開始位置・長さ・パート）
    r = subprocess.run([sys.executable, str(ROOT / "run_eval.py"), str(stems), "--max-seconds", "8"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    meta = read_hits_csv_meta(stems / "s01" / "hits_pseudo.csv")
    assert set(meta) == {"start_sec", "seconds", "parts"} and meta["parts"] == "hihat,kick,snare"
    assert float(meta["seconds"]) <= 8.0 + 1e-6
    songs = read_stem_songs(stems)
    assert [s.song_id for s in songs] == ["s01"]
    s = songs[0]
    assert s.truth == "pseudo" and s.labeled == ("kick", "snare", "hihat") and abs(s.start - float(meta["start_sec"])) < 1e-6
    assert s.seconds == float(meta["seconds"]) and all(0 <= h.time < s.seconds for h in s.hits) and len(s.hits) > 10
    # 区間のデータセット: 形とマスク（タム・シンバルのラベルは無い）
    cfg = FeatureConfig()
    ds = StemChunks(songs, cfg, chunk_sec=2.0, chunks_per_song=3, do_augment=True)
    assert len(ds) == 3
    feat, y, mask = ds[0]
    assert feat.shape[1] == cfg.n_mels and y.shape == (feat.shape[0], 5) and tuple(mask.tolist()) == (1.0, 1.0, 1.0, 0.0, 0.0)
    assert y[:, :3].sum() > 0 and y[:, 3:].sum() == 0
    # 古い CSV（コメント行なし）でも、run_eval と同じ区間になる
    hits = load_hits(stems / "s01" / "hits_pseudo.csv")
    save_hits_csv(hits, stems / "s01" / "hits_pseudo.csv")
    assert read_hits_csv_meta(stems / "s01" / "hits_pseudo.csv") == {}
    old = read_stem_songs(stems, max_seconds=8.0)[0]
    assert abs(old.start - s.start) < 1e-6 and abs(old.seconds - s.seconds) < 0.05 and old.labeled == s.labeled


def test_resolve_start_prefers_start_txt(tmp_path):
    song = tmp_path / "s01"
    _make_song(song, _rock_pattern(4), lead_in=4.0)
    import soundfile as sf

    x, sr = sf.read(song / "drum.wav")
    assert 3.0 <= resolve_start(song, x, sr) <= 5.0  # キック＋スネアの合計で自動検出
    assert resolve_start(song, x, sr, start_sec=1.5) == 1.5  # --start
    (song / "start.txt").write_text("2.0", encoding="utf-8")
    assert resolve_start(song, x, sr, start_sec=1.5) == 2.0  # start.txt が最優先


def test_finetune_leave_one_out(tmp_path):
    stems = tmp_path / "stems"
    rng = np.random.default_rng(0)
    for i in range(3):
        ref = _rock_pattern(6, bpm=float(rng.uniform(95, 130)))
        _make_song(stems / f"s{i + 1:02d}", ref)
    r = subprocess.run([sys.executable, str(ROOT / "run_eval.py"), str(stems), "--max-seconds", "10"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cfg = FeatureConfig()
    ckpt = tmp_path / "init.pt"
    _init_checkpoint(ckpt, cfg)
    out = tmp_path / "models" / "ft.onnx"
    r = subprocess.run([sys.executable, str(ROOT / "train_transcriber.py"), "--finetune-stems", str(stems), "--init-from", str(ckpt),
                        "--holdout", "s02,s03", "--epochs", "1", "--batch", "2", "--seconds", "3", "--chunks-per-song", "2", "--workers", "0",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for h in ("s02", "s03"):
        m = out.with_name(f"ft_{h}.onnx")
        assert m.exists() and m.with_suffix(".json").exists() and m.with_suffix(".pt").exists()
        import json

        meta = json.loads(m.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["holdout"] == h and h not in meta["train_songs"] and len(meta["train_songs"]) == 2
        assert "evaluation only" in meta["trained_on"] and set(meta["thresholds"]) == set(CLASSES5)
    summary = (out.with_name("ft_loo.md")).read_text(encoding="utf-8")
    assert "| s02 |" in summary and "| s03 |" in summary and "| 平均 |" in summary
    assert "ラベルのずれの見積もり" in r.stdout
    # 曲ごとのモデルを run_eval で使う（{id} の置き換え。モデルの無い s01 は飛ばす）
    r = subprocess.run([sys.executable, str(ROOT / "run_eval.py"), str(stems), "--max-seconds", "10", "--model", str(out.with_name("ft_{id}.onnx")),
                        "--out", str(tmp_path / "r.md")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "| s02 |" in md and "| s03 |" in md and "| s01 |" not in md and "ft_s02" in md
    assert "s01: モデルが無い" in r.stderr


def test_finetune_with_egmd_mix(tmp_path):
    """--egmd を併せて渡すと、E-GMD の区間（全種類ラベルあり）を混ぜ、E-GMD の検証の前後も出す。"""
    from tests.test_egmd import _fake_egmd

    stems = tmp_path / "stems"
    for i in range(2):
        _make_song(stems / f"s{i + 1:02d}", _rock_pattern(5, bpm=100.0 + 20 * i))
    r = subprocess.run([sys.executable, str(ROOT / "run_eval.py"), str(stems), "--max-seconds", "8"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    egmd = tmp_path / "e-gmd"
    _fake_egmd(egmd, n=3, seconds=8.0)
    ckpt = tmp_path / "init.pt"
    _init_checkpoint(ckpt, FeatureConfig(), old_format=True)  # しきい値を持たない古い形式（横に json も無い）
    out = tmp_path / "ft.onnx"
    r = subprocess.run([sys.executable, str(ROOT / "train_transcriber.py"), "--finetune-stems", str(stems), "--init-from", str(ckpt),
                        "--egmd", str(egmd), "--max-tracks", "2", "--val-tracks", "1", "--val-seconds", "5", "--chunks-per-track", "1",
                        "--epochs", "1", "--batch", "2", "--seconds", "3", "--chunks-per-song", "2", "--workers", "0", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "E-GMD の混ぜ込み 2 本" in r.stdout and "E-GMD 検証 M0-1" in r.stdout
    assert out.exists() and (tmp_path / "ft_fit.md").exists()  # holdout 無し: 当てはまりの表
    assert "（学習曲の当てはまり）" in (tmp_path / "ft_fit.md").read_text(encoding="utf-8")
