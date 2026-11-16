"""解析の参照実装（計画書のモジュール `Analyzer`）: ドラムのステム 1 本 → キットと打点。

Phase 1 段階 1（2026-11-16 合格）で確かめた製品の経路を、1 つの関数にまとめたもの。段階 2 で C++／ONNX に移すときの基準になる。

    ② 採譜          ONNX の採譜モデルで打点（時刻・種類）を出す。`hits` を渡せばそれを使う
    ③ パート分け    打点で縛った NMFD（`nmfd.py`）。タムの打が 4 つ以上あればタムのパートも作る
    （打点の合わせ直し）各打の時刻を、そのパートの音の立ち上がりに合わせ直す
    ④ 一打の切り出し 同じ種類の前後に打の無い打を、次の同じ種類の打の手前まで切り出す
    ⑤ キット         種類ごとに、特徴（音量・減衰・明るさ・帯域）の k-means の代表 `kit_size` 音
    （強さ）         各打の音量を、種類ごとの範囲（`Kit.level_range`）で MIDI の強さ 1〜127 に直す

返り値の `Analysis.hits` と `Analysis.kit` があれば、MIDI だけで元に近い音を鳴らせる（`extract.render_midi`、プラグインの `SampleEngine`）。
テンポ・拍（①）とエフェクトの当てはめ（⑥）はまだ入っていない。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .extract import (
    Kit, build_kit_by_features, cut_segments_from_parts, hit_segments_from_parts, level_range_for, refine_onsets, render_midi, save_kit,
    segment_features, silent_hits, velocity_for_level,
)
from .labels import COARSE_CLASSES, Hit, save_hits_midi, to_coarse
from .nmfd import config_for, parts_for_hits, separate


@dataclass
class Analysis:
    sr: int
    duration: float  # 解析した長さ（秒）
    hits: list[Hit]  # 合わせ直した時刻、粗い種類、強さ（`kit.level_range` の約束）
    levels_db: list[float | None]  # 各打のピーク（dB）。`hits` と同じ順
    kit: Kit
    parts: dict[str, np.ndarray]  # パート別の音（"residual" を含む）
    timings: dict[str, float] = field(default_factory=dict)  # 段ごとの所要時間（秒）
    model: str = ""
    n_dropped: int = 0  # そのパートに音が無くて除いた打の数


def analyze(x: np.ndarray, sr: int, model: str | Path | None = None, hits: list[Hit] | None = None, threshold: float | None = None,
            kit_size: int = 24, nmfd_options: dict | None = None) -> Analysis:
    """ドラムのステム（モノラル）を解析する。`model`（ONNX の採譜モデル）か `hits`（打点）のどちらかを渡す。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if hits is None:
        if model is None:
            raise ValueError("model（ONNX の採譜モデル）か hits（打点）のどちらかが要る")
        from .infer import transcribe_onnx

        hits = transcribe_onnx(x, sr, model, threshold)
    hits = sorted((Hit(h.time, to_coarse(h.cls), None) for h in hits if 0.0 <= h.time < x.size / sr and to_coarse(h.cls) in COARSE_CLASSES),
                  key=lambda h: h.time)
    timings["採譜"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    part_names = parts_for_hits(hits)
    sep = separate(x, sr, hits, config_for(sr, **(nmfd_options or {})), parts=part_names) if hits else {}
    part_audio = {c: (sep[c], sr) for c in part_names if c in sep}
    timings["パート分け"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = refine_onsets(part_audio, [h for h in hits if h.cls in part_audio])
    hits.sort(key=lambda h: h.time)

    def own_levels(hs: list[Hit]) -> list[float | None]:
        return [20.0 * np.log10(np.max(np.abs(seg)) + 1e-9) if seg is not None else None for seg in hit_segments_from_parts(part_audio, hs)]

    levels = own_levels(hits)
    drop = silent_hits(hits, levels)  # そのパートに音の無い打（誤検出）を除く。除くと、前の打の区間が次の打まで伸びる
    n_dropped = int(sum(drop))
    if n_dropped:
        hits = [h for h, d in zip(hits, drop) if not d]
        levels = own_levels(hits)
    segs = cut_segments_from_parts(part_audio, hits)
    kit = build_kit_by_features(segs, sr, n_per_class=kit_size, feature_fn=segment_features)
    for c in kit.parts:
        kit.level_range[c] = level_range_for([lv for lv, h in zip(levels, hits) if lv is not None and h.cls == c])
    hits = [Hit(h.time, h.cls, velocity_for_level(lv, kit.level_range[h.cls]) if lv is not None and h.cls in kit.level_range else 100)
            for h, lv in zip(hits, levels)]
    timings["切り出しとキット"] = time.perf_counter() - t0
    return Analysis(sr, x.size / sr, hits, levels, kit, sep, timings, str(model) if model else "", n_dropped)


def save_analysis(a: Analysis, out_dir: str | Path, offset_sec: float = 0.0) -> dict:
    """キット（`kit.json` と WAV）、打点（`hits.mid`）、まとめ（`analysis.json`）を書く。返り値は `analysis.json` の中身。

    `offset_sec` は、解析した区間がファイルの先頭から何秒目に始まるか。MIDI の時刻に足して、元のステムと頭をそろえる。
    """
    out = Path(out_dir)
    save_kit(a.kit, out)
    save_hits_midi(a.hits, out / "hits.mid", offset_sec=offset_sec)
    counts = {c: sum(1 for h in a.hits if h.cls == c) for c in COARSE_CLASSES}
    meta = {
        "sr": a.sr, "seconds": a.duration, "offset_sec": offset_sec, "model": Path(a.model).name if a.model else "",
        "hits": {c: n for c, n in counts.items() if n},
        "dropped_silent_hits": a.n_dropped,
        "kit": {c: a.kit.n_samples(c) for c in a.kit.parts},
        "level_range_db": {c: [round(lo, 2), round(hi, 2)] for c, (lo, hi) in a.kit.level_range.items()},
        "timings_sec": {k: round(v, 2) for k, v in a.timings.items()},
    }
    (out / "analysis.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def preview(a: Analysis) -> np.ndarray:
    """MIDI だけで鳴らした音（プラグインで鳴る音の予想）。"""
    return render_midi(a.hits, a.kit, duration=a.duration)
