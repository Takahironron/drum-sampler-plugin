"""一打の切り出しと強さ別の層（計画書 3.4 の候補 (a) の最小版）。

手順:
1. 採譜の結果（打点ラベル）から「前後に他の打が重ならない打」を選ぶ（`select_isolated_hits`）。
   前 `pre_sec`（既定 0.1 秒。直前の打の尾が混ざらないように）以内と後 `post_sec`（既定 0.15 秒）以内に、種類を問わず他の打が無いもの。
2. 打点の直前から次の打の直前まで（上限 `max_len_sec`）を切り出し、短いフェードを付ける（`cut_segments`）。
3. 種類ごとにピークの大きさで並べ、強さ別の層に分ける。同じ層の複数の打はラウンドロビンにする（`build_kit`）。
4. 切り出したキットで打点ラベルを鳴らし直す（`render_kit`）。元のステムとの STFT 距離が「信頼度」の材料になる。

音の尾は次の打で切れるので、延長（モーダル合成か減衰の当てはめ）は Phase 2 の課題。ここではやらない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .labels import COARSE_CLASSES, Hit, to_coarse


@dataclass
class Segment:
    cls: str  # 粗い種類
    time: float  # 元の打点（秒）
    audio: np.ndarray  # モノラル
    peak_db: float
    confidence: float | None = None  # 分離の確からしさ（`nmfd.hit_confidence`。分からなければ None）


@dataclass
class Layer:
    vel_lo: int
    vel_hi: int
    samples: list[np.ndarray] = field(default_factory=list)  # ラウンドロビン
    source_times: list[float] = field(default_factory=list)
    peaks_db: list[float] = field(default_factory=list)  # 各音のピーク（音量合わせ用）
    features: list[list[float]] = field(default_factory=list)  # 各音の特徴（近い音を選ぶ用。無ければ空）
    confidences: list[float | None] = field(default_factory=list)  # 各音の分離の確からしさ（分からなければ空か None）


@dataclass
class Kit:
    sr: int
    parts: dict[str, list[Layer]]  # 種類 → 弱い層から強い層へ
    feat_norm: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)  # 種類 → (平均, 標準偏差)。特徴の正規化
    # 種類 → (下限 dB, 上限 dB)。MIDI の強さ 1〜127 が、打のピークのこの範囲に dB で線形に対応する（`velocity_for_level`）。
    # MIDI だけでキットを鳴らすとき（プラグイン、`render_midi`）に、元の打の音量を再現するための約束
    level_range: dict[str, tuple[float, float]] = field(default_factory=dict)

    def n_samples(self, cls: str) -> int:
        return sum(len(l.samples) for l in self.parts.get(cls, []))

    def all_samples(self, cls: str) -> list[tuple[np.ndarray, float, list[float] | None]]:
        out = []
        for l in self.parts.get(cls, []):
            for j, smp in enumerate(l.samples):
                peak = l.peaks_db[j] if j < len(l.peaks_db) else 20.0 * np.log10(np.max(np.abs(smp)) + 1e-9)
                feat = l.features[j] if j < len(l.features) else None
                out.append((smp, peak, feat))
        return out

    def pick_nearest(self, cls: str, feature: np.ndarray, exclude_time: float | None = None) -> tuple[np.ndarray, float] | None:
        """特徴の近い音を返す（音, ピーク dB）。特徴の無いキットなら None。

        `exclude_time` を渡すと、その時刻（±5 ms）に由来する音を候補から外す（自分自身を除く上限の測定用）。
        """
        cands = []
        for l in self.parts.get(cls, []):
            for j, smp in enumerate(l.samples):
                if j >= len(l.features):
                    continue
                if exclude_time is not None and j < len(l.source_times) and abs(l.source_times[j] - exclude_time) < 0.005:
                    continue
                peak = l.peaks_db[j] if j < len(l.peaks_db) else 20.0 * np.log10(np.max(np.abs(smp)) + 1e-9)
                cands.append((smp, peak, l.features[j]))
        if not cands or cls not in self.feat_norm:
            return None
        mean, std = (np.asarray(v, dtype=float) for v in self.feat_norm[cls])
        q = (np.asarray(feature, dtype=float) - mean) / std
        d = [np.sum(((np.asarray(f) - mean) / std - q) ** 2) for _, _, f in cands]
        smp, peak, _ = cands[int(np.argmin(d))]
        return smp, peak


CHOKE_CLASSES = {"kick", "snare", "hihat", "tom"}  # 同じ種類の次の打で音を止める（シンバルは重ねる）

# パート別の切り出しで、一打の区間の上限（秒）。シンバルは余韻が長い（クラッシュは 3〜6 秒鳴る）ので長く取る。
# 全種類 1.5 秒で切ると、シンバルの余韻のぶん高域が減る
MAX_LEN_SEC: dict[str, float] = {"kick": 1.5, "snare": 1.5, "hihat": 1.5, "tom": 2.0, "cymbal": 4.0}


def max_len_for(cls: str, max_len_sec: float | dict[str, float] | None = None) -> float:
    """種類ごとの区間の上限。`max_len_sec` が None なら `MAX_LEN_SEC`、辞書ならその種類の値、数ならその値。"""
    if max_len_sec is None:
        return MAX_LEN_SEC.get(cls, 1.5)
    if isinstance(max_len_sec, dict):
        return float(max_len_sec.get(cls, MAX_LEN_SEC.get(cls, 1.5)))
    return float(max_len_sec)


SAMPLE_LEAD_SEC = 0.002  # 切り出した音は打点のこの秒数だけ前から始まる（`cut_segments` の lead_sec）
VELOCITY_RANGE_DB = 30.0  # 強さ 1〜127 に対応させる音量の幅の下限（dB）。最も大きい打からこの幅までは必ず表せる


def level_range_for(levels_db: list[float] | np.ndarray) -> tuple[float, float]:
    """打のピーク（dB）の並びから、強さ 1〜127 に対応させる (下限, 上限) を決める。

    上限は最も大きい打。下限は「上限 − 30 dB」と最も小さい打の小さい方（どの打も強さで表せる。打が全部同じ大きさの
    打ち込みの曲でも、DAW で強さを下げれば小さく鳴る）。
    """
    v = np.asarray([x for x in levels_db if x is not None], dtype=float)
    if v.size == 0:
        return (-VELOCITY_RANGE_DB, 0.0)
    hi = float(v.max())
    return (float(min(v.min(), hi - VELOCITY_RANGE_DB)), hi)


def silent_hits(hits: list[Hit], levels_db: list[float | None], floor_db: float = -40.0) -> list[bool]:
    """「音の無い打」の印（`hits` と同じ順の真偽）。その種類の大きい打（上位 5%）より `floor_db` 以上小さい打は、鳴らしても聞こえない。

    採譜の誤検出のうち、そのパートに音が無い時刻のもの（パート分けの後で分かる）がこれに当たる。MIDI とキットから除く。
    """
    out = [False] * len(hits)
    by_cls: dict[str, list[int]] = {}
    for i, h in enumerate(hits):
        by_cls.setdefault(to_coarse(h.cls), []).append(i)
    for idx in by_cls.values():
        vals = [levels_db[i] for i in idx if levels_db[i] is not None]
        if not vals:
            for i in idx:
                out[i] = True
            continue
        ref = float(np.percentile(vals, 95))
        for i in idx:
            out[i] = levels_db[i] is None or bool(levels_db[i] < ref + floor_db)
    return out


def velocity_for_level(level_db: float, level_range: tuple[float, float]) -> int:
    """打のピーク（dB）→ MIDI の強さ（1〜127）。範囲の中で dB に線形。"""
    lo, hi = level_range
    return int(round(1 + 126 * float(np.clip((level_db - lo) / max(hi - lo, 1e-6), 0.0, 1.0))))


def level_for_velocity(velocity: int | None, level_range: tuple[float, float]) -> float:
    """MIDI の強さ → 目標のピーク（dB）。`velocity_for_level` の逆。強さが無ければ上限。"""
    lo, hi = level_range
    if velocity is None:
        return hi
    return lo + (hi - lo) * (min(max(int(velocity), 1), 127) - 1) / 126.0


def _refine_onset_single(x: np.ndarray, sr: int, time: float, search_sec: float) -> float | None:
    """1 打だけで立ち上がりを探す: 打点の前後 `search_sec` の中で、「その後 3 ms の大きさ ÷ その前 6 ms の大きさ」が最大の位置。

    はっきりした立ち上がりが無ければ（比が振幅で 1.5 未満）None。打の少ない種類（ひな形を作れない）にだけ使う。前の音の余韻や、
    分離の窓による前ぶれ（プリエコー）、低い音の包絡の波打ちがあると、数 ms〜十数 ms 離れた位置を取ることがある（StemGMD の正解の打点で、
    キックの 11% が 5 ms 以上ずれた）ので、打が 4 つ以上ある種類は `refine_onsets` がひな形に合わせる。
    """
    n_after, n_before = max(int(0.003 * sr), 1), max(int(0.006 * sr), 1)
    c = int(round(time * sr))
    lo, hi = max(c - int(search_sec * sr), n_before), min(c + int(search_sec * sr), x.size - n_after)
    if hi <= lo:
        return None
    p = np.asarray(x[lo - n_before : hi + n_after], dtype=np.float64) ** 2
    cum = np.concatenate([[0.0], np.cumsum(p)])
    i = np.arange(n_before, n_before + (hi - lo))
    after = (cum[i + n_after] - cum[i]) / n_after
    before = (cum[i] - cum[i - n_before]) / n_before
    floor = 1e-4 * float(after.max()) + 1e-12  # 無音どうしの比で動かないように、その打の大きさに対する小さな床を足す
    ratio = (after + floor) / (before + floor)
    k = int(np.argmax(ratio))
    if ratio[k] < 1.5**2:
        return None
    return (lo + k) / sr


ONSET_ENV_STEP_SEC = 0.00025  # 立ち上がりを合わせるときの包絡の刻み（0.25 ms）


def _onset_envelope(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """1 ms の二乗平均の包絡を 0.25 ms 刻みで。返り値は（包絡, 1 秒あたりの点の数）。"""
    x = np.asarray(x, dtype=np.float64)
    q = max(int(round(ONSET_ENV_STEP_SEC * sr)), 1)
    w = max(int(0.001 * sr), 1)
    cum = np.concatenate([[0.0], np.cumsum(x * x)])
    centers = np.arange(0, x.size, q)
    env = np.sqrt((cum[np.minimum(centers + w // 2 + 1, x.size)] - cum[np.maximum(centers - w // 2, 0)]) / w)
    return env, sr / q


def _rise_strength(env: np.ndarray, rate: float) -> np.ndarray:
    """立ち上がりの強さ: 包絡が 2 ms 前より増えた分（減った所は 0）。"""
    lag = max(int(0.002 * rate), 1)
    return np.maximum(env - np.concatenate([np.zeros(lag), env[:-lag]]), 0.0)


def refine_onsets(parts: dict[str, tuple[np.ndarray, int]], hits: list[Hit], search_sec: float = 0.02, iterations: int = 3,
                  rise: float = 0.1, post_sec: float = 0.02, shape: str = "rise", stay: float = 0.2,
                  stats: dict | None = None) -> list[Hit]:
    """各打の時刻を、そのパートの音の立ち上がりに合わせ直す（採譜の時刻は約 10 ms 刻みで、最大 1 フレームずれる）。

    種類ごとに「立ち上がりの形のひな形」を作り、各打をひな形との相互相関が最大になる位置へ動かす。

    - 比べるのは、立ち上がりの強さ（1 ms の二乗平均の包絡が、2 ms 前より増えた分。0.25 ms 刻み）。`shape="env"` なら包絡そのもの。
      包絡そのものだと、余韻の長さが違う音（ハイハットの開閉）で、音の大きい部分どうしが重なる位置に引かれる。
    - ひな形は、各打の前 30 ms〜後 `post_sec` を、その中の最大で割り、打をまたいで中央値を取ったもの。
    - 動かす幅は、元の時刻から ±`search_sec` と、隣の同じ種類の打との間隔の 0.45 倍の小さい方まで。比べる区間も、隣の同じ種類の
      打の手前までに限る（装飾音や連打で、隣の打の立ち上がりに引かれないように）。元の時刻から離れるほど、相関を少し割り引く
      （`stay`。±`search_sec` の端で 1 − stay 倍。立ち上がりのはっきりしない小さい打を、遠くへ動かさない）。
    - ひな形を作り直しながら `iterations` 回繰り返し、最後に包絡のひな形の立ち上がり（山の手前で、谷から山までの `rise` の高さを
      上向きに超える位置）を打点にする。正解の時刻と比べると、キックは 0.6〜0.9 ms 後ろ（ゆっくり立ち上がる）、スネアは ±0.5 ms、
      ハイハットは 1.3〜3.0 ms 手前（分離の窓による前ぶれが先に上がる）。種類の中では一定なので、音の位置は変わらない
      （切り出しは打点の `SAMPLE_LEAD_SEC` 前から）。

    同じ種類の打は立ち上がりの形が重なるように時刻が決まるので、切り出した音の頭から立ち上がりまでの長さがそろい、どの音を
    どの打の時刻に置いても位置が合う。MIDI だけで鳴らすときに必要（方式 C は打ごとに相互相関で置き直すので、ずれが隠れていた。
    区分 A の a07 で「MIDI だけだとキックが悪い」と聞こえた原因。2026-11-17）。

    1 打ごとに立ち上がりを探す従来の方法（`_refine_onset_single`）との比較（正解の時刻との差のばらつきが 5 ms を超える打の割合。
    正解の打点／取りこぼしと時刻の揺れを入れた打点）: StemGMD キック 11%／9% → 1%／1%、スネア 7%／4% → 2%／2%、ハイハット 9%／11% → 5%／8%。
    E-GMD キック 4%／5% → 1%／1%、スネア 9%／5% → 8%／6%、ハイハット 15%／16% → 8%／10%。残るのは、小さい打（ゴーストノート）と、
    隣の打が 60 ms 以内にある打、スネアと同時に鳴るハイハット。ひな形との相関が 0.9 以上の打（キックの 9 割、スネアとハイハットの 5〜8 割）は、
    ずれるのが 1.4% 以下。
    打が 4 つ未満の種類は従来の方法で合わせる。パートの無い種類、端の打、音の無い打は動かさない。

    `stats`（空の dict を渡す）には、打ごとの `"corr"`（ひな形との相関の最大値、0〜1。合わせなかった打は None）を `hits` と同じ順で入れる。
    """
    out = list(hits)
    corr_out: list[float | None] = [None] * len(hits)
    by_cls: dict[str, list[int]] = {}
    for i, h in enumerate(hits):
        by_cls.setdefault(to_coarse(h.cls), []).append(i)
    for cls, idx in by_cls.items():
        if cls not in parts:
            continue
        x, sr = parts[cls]
        x = np.asarray(x, dtype=np.float64)
        if len(idx) < 4:
            for i in idx:
                t = _refine_onset_single(x, sr, hits[i].time, search_sec)
                if t is not None:
                    out[i] = Hit(t, hits[i].cls, hits[i].velocity)
            continue
        env, rate = _onset_envelope(x, sr)  # 1 ms の二乗平均を 0.25 ms 刻みで
        pre, post, span = int(0.03 * rate), int(post_sec * rate), int(search_sec * rate)
        env_mark = env  # 立ち上がりの位置は、いつも包絡そのもので決める
        if shape == "rise":  # 立ち上がりの強さ（2 ms 前との差の正の部分）
            env = _rise_strength(env, rate)
        cum2 = np.concatenate([[0.0], np.cumsum(env * env)])
        times = np.array([hits[i].time for i in idx])
        order = np.argsort(times, kind="stable")
        d = np.diff(times[order])
        gap_prev = np.full(len(idx), np.inf)
        gap_next = np.full(len(idx), np.inf)
        gap_prev[order[1:]] = d
        gap_next[order[:-1]] = d
        max_lag = np.minimum(span, np.floor(0.45 * np.minimum(gap_prev, gap_next) * rate)).astype(int)
        margin = int(0.002 * rate)
        n_pre = np.clip(np.floor(gap_prev * rate) - margin, 0, pre).astype(int)  # 比べる区間（隣の同じ種類の打の手前まで）
        n_post = np.clip(np.floor(gap_next * rate) - margin, 0, post).astype(int)
        orig = np.round(times * rate).astype(int)
        cur = orig.copy()
        frac = np.zeros(len(idx))  # 0.25 ms より細かい分（放物線で補う）
        aligned = np.zeros(len(idx), dtype=bool)
        corr = np.zeros(len(idx))  # ひな形との相関（元の時刻からの遠さで割り引く前の値）

        def template(positions: np.ndarray) -> np.ndarray | None:
            rows = []
            for c in positions:
                if c - pre < 0 or c + post > env.size:
                    continue
                seg = env[c - pre : c + post]
                m = float(seg.max())
                if m > 1e-9:
                    rows.append(seg / m)
            if len(rows) < 4:
                return None
            t = np.median(np.stack(rows), axis=0)
            return t - t.min()

        peaks = np.array([float(env[max(c - pre, 0) : max(c + post, 1)].max()) for c in orig])
        ref_level = float(np.median(peaks))
        tmpl = template(orig)
        for _ in range(iterations):
            if tmpl is None:
                break
            for j in range(len(idx)):
                ml = int(max_lag[j])
                lo_t, hi_t = pre - int(n_pre[j]), pre + int(n_post[j])
                a, b = orig[j] - ml - int(n_pre[j]), orig[j] + ml + int(n_post[j])
                if ml <= 0 or hi_t - lo_t < int(0.008 * rate) or a < 0 or b > env.size or peaks[j] < 0.01 * ref_level:
                    continue  # 隣が近すぎる打、端の打、音の無い打は動かさない
                t_j = tmpl[lo_t:hi_t]
                if not np.any(t_j > 0):
                    continue
                num = np.correlate(env[a:b], t_j, mode="valid")  # 長さ 2·ml + 1。k 番目は元の時刻からのずれ k − ml
                k0 = np.arange(2 * ml + 1) + a
                v = num / ((np.sqrt(cum2[k0 + t_j.size] - cum2[k0]) + 1e-12) * (float(np.linalg.norm(t_j)) + 1e-12))
                raw = v
                if stay > 0:  # 元の時刻から離れるほど少し不利にする（立ち上がりのはっきりしない打を、遠くへ動かさない）
                    v = v * (1.0 - stay * np.abs(np.arange(-ml, ml + 1)) / max(span, 1))
                k = int(np.argmax(v))
                corr[j] = float(raw[k])
                f = 0.0
                if 0 < k < v.size - 1:
                    den = v[k - 1] - 2 * v[k] + v[k + 1]
                    if den < 0:
                        f = float(np.clip(0.5 * (v[k - 1] - v[k + 1]) / den, -0.5, 0.5))
                cur[j], frac[j], aligned[j] = orig[j] + k - ml, f, True
            if aligned.sum() < 4:
                tmpl = None
                break
            tmpl = template(cur[aligned])
        if tmpl is None:
            continue
        if shape != "env":
            env = env_mark
            tmpl = template(cur[aligned])
            if tmpl is None:
                continue
        # ひな形の立ち上がりの位置: 打点の近くの山の手前で、谷から山までの rise の高さを上向きに超える位置。
        # 低い音は包絡が周期ごとに落ち込む（キックの 50〜80 Hz）ので、「3 ms 以上続けて下回った後」の超え方だけを立ち上がりとみなす
        lo_p, hi_p = max(pre - span, 1), min(pre + span + int(0.015 * rate), tmpl.size)
        k_peak = lo_p + int(np.argmax(tmpl[lo_p:hi_p]))
        k_lo = max(k_peak - int(0.03 * rate), 0)
        valley = float(tmpl[k_lo : k_peak + 1].min())
        level = valley + rise * (float(tmpl[k_peak]) - valley)
        hold = max(int(0.003 * rate), 1)
        below = tmpl[k_lo : k_peak + 1] < level
        k, run, first = None, 0, None
        for m in range(below.size):
            if below[m]:
                run += 1
                continue
            if m > 0 and below[m - 1]:  # 上向きに超えた
                if first is None:
                    first = m
                if run >= hold or m - run == 0:  # 十分長く下回っていたか、窓の頭から下回っていた
                    k = m
            run = 0
        if k is None:
            k = first if first is not None else 0
        k += k_lo
        f = 0.0
        if k > 0 and tmpl[k] > tmpl[k - 1]:  # しきい値を超える位置を、前の点との間で直線で補う
            f = float(np.clip((level - tmpl[k - 1]) / (tmpl[k] - tmpl[k - 1]), 0.0, 1.0)) - 1.0
        shift = k + f - pre
        for j, i in enumerate(idx):
            if aligned[j]:  # 合わせなかった打は元の時刻のまま
                out[i] = Hit(float(max((cur[j] + frac[j] + shift) / rate, 0.0)), hits[i].cls, hits[i].velocity)
                corr_out[i] = float(corr[j])
    if stats is not None:
        stats["corr"] = corr_out
    return out


def select_isolated_hits(hits: list[Hit], pre_sec: float = 0.1, post_sec: float = 0.15) -> list[tuple[Hit, float]]:
    """重ならない打と、その打に使える長さ（次の打までの秒数）を返す。"""
    hs = sorted(hits, key=lambda h: h.time)
    out: list[tuple[Hit, float]] = []
    for i, h in enumerate(hs):
        prev_t = hs[i - 1].time if i > 0 else -np.inf
        next_t = hs[i + 1].time if i + 1 < len(hs) else np.inf
        if h.time - prev_t < pre_sec or next_t - h.time < post_sec:
            continue
        out.append((h, float(next_t - h.time)))
    return out


def cut_segments(
    x: np.ndarray,
    sr: int,
    hits: list[Hit],
    pre_sec: float = 0.1,
    post_sec: float = 0.15,
    max_len_sec: float = 1.5,
    lead_sec: float = SAMPLE_LEAD_SEC,
    fade_out_sec: float = 0.01,
) -> list[Segment]:
    x = np.asarray(x, dtype=np.float64).ravel()
    segs: list[Segment] = []
    for h, avail in select_isolated_hits(hits, pre_sec, post_sec):
        start = max(int(round((h.time - lead_sec) * sr)), 0)
        length = int(min(avail - lead_sec, max_len_sec) * sr)
        end = min(start + length, x.size)
        if end - start < int(0.01 * sr):
            continue
        a = x[start:end].copy()
        n_in = max(int(0.001 * sr), 1)
        a[:n_in] *= np.linspace(0.0, 1.0, n_in)
        n_out = min(int(fade_out_sec * sr), a.size)
        a[-n_out:] *= np.linspace(1.0, 0.0, n_out)
        peak = float(np.max(np.abs(a)) + 1e-9)
        segs.append(Segment(to_coarse(h.cls), h.time, a, 20.0 * np.log10(peak)))
    return segs


def cut_segments_from_parts(
    parts: dict[str, tuple[np.ndarray, int]],
    hits: list[Hit],
    pre_sec: float = 0.1,
    post_sec: float = 0.15,
    max_len_sec: float | dict[str, float] | None = None,
) -> list[Segment]:
    """パート別に分離された音（種類 → (波形, sr)）から切り出す。

    他の種類の打は別のステムにあるので、重なりの判定は同じ種類の打どうしだけで行う。
    混ざったドラムから切るより、切り出せる打がずっと多くなる。区間の上限は種類ごと（`MAX_LEN_SEC`）。
    """
    segs: list[Segment] = []
    for cls, (x, sr) in parts.items():
        own = [h for h in hits if to_coarse(h.cls) == cls]
        segs += cut_segments(x, sr, own, pre_sec=pre_sec, post_sec=post_sec, max_len_sec=max_len_for(cls, max_len_sec))
    return segs


def self_render_from_parts(
    parts: dict[str, tuple[np.ndarray, int]],
    hits: list[Hit],
    duration: float,
    max_len_sec: float | dict[str, float] | None = None,
    fade_out_sec: float = 0.01,
) -> np.ndarray:
    """各打を「その打自身の区間」（打点から同じ種類の次の打まで）で鳴らし直す（自己再現）。

    層やラウンドロビンの量子化が無いので、元との距離は「打点の誤り・取りこぼし」と「区間の切り方」だけを映す。
    キットでの再現との差が、層とラウンドロビンで失う分。区間は、キットの音と同じく打点の `SAMPLE_LEAD_SEC` 前から取る
    （打点は立ち上がりの頭より少し後ろに来ることがある。打点ちょうどで切ると、前の区間のフェードが立ち上がりの頭に掛かる）。
    """
    sr = next(iter(parts.values()))[1]
    n = int(duration * sr)
    out = np.zeros(n)
    for cls, (x, psr) in parts.items():
        own = sorted((h for h in hits if to_coarse(h.cls) == cls), key=lambda h: h.time)
        limit = max_len_for(cls, max_len_sec)
        for i, h in enumerate(own):
            start = max(int(round((h.time - SAMPLE_LEAD_SEC) * psr)), 0)
            next_t = own[i + 1].time if i + 1 < len(own) else h.time + limit
            end = min(start + int(min(next_t - h.time, limit) * psr), x.size, n)
            if end - start < int(0.005 * psr):
                continue
            a = x[start:end].copy()
            n_out = min(int(fade_out_sec * psr), a.size)
            a[-n_out:] *= np.linspace(1.0, 0.0, n_out)
            out[start:end] += a
    return out


FEATURE_NAMES = ("peak_db", "log_decay", "log_centroid", "low_db", "mid_db", "high_db")


def segment_features(a: np.ndarray, sr: int) -> np.ndarray:
    """一打の特徴: [ピーク dB, log10(減衰時間 秒), log(スペクトル重心 Hz), 低域 dB, 中域 dB, 高域 dB]。

    減衰時間は 5 ms 窓の RMS が最大から 20 dB 下がるまでの時間。重心は最初の 30 ms のスペクトル。
    帯域のエネルギー（低 <150 Hz、中 150〜2 kHz、高 >5 kHz）は最初の 60 ms の RMS（dB）で、キックの胴鳴りの量や
    スネアの響き線の量の違いを見る。
    """
    a = np.asarray(a, dtype=np.float64)
    peak = float(np.max(np.abs(a)) + 1e-9)
    win = max(int(0.005 * sr), 1)
    n = a.size // win
    if n >= 1:
        env = np.sqrt(np.mean(a[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
        env_db = 20.0 * np.log10(env)
        i_max = int(np.argmax(env_db))
        below = np.flatnonzero(env_db[i_max:] < env_db[i_max] - 20.0)
        decay = (i_max + below[0]) * win / sr if below.size else a.size / sr
    else:
        decay = a.size / sr
    head = a[: max(int(0.03 * sr), 64)]
    spec = np.abs(np.fft.rfft(head * np.hanning(head.size)))
    f = np.fft.rfftfreq(head.size, 1.0 / sr)
    centroid = float(np.sum(f * spec) / (np.sum(spec) + 1e-12)) + 1.0
    body = a[: max(int(0.06 * sr), 128)]
    spec_b = np.abs(np.fft.rfft(body * np.hanning(body.size))) ** 2
    f_b = np.fft.rfftfreq(body.size, 1.0 / sr)
    bands = []
    for lo, hi in ((20.0, 150.0), (150.0, 2000.0), (5000.0, sr / 2)):
        sel = (f_b >= lo) & (f_b < hi)
        bands.append(10.0 * np.log10(np.mean(spec_b[sel]) + 1e-12) if np.any(sel) else -120.0)
    return np.array([20.0 * np.log10(peak), np.log10(decay + 1e-3), np.log(centroid), *bands])


def head_embedding(a: np.ndarray, sr: int, head_sec: float = 0.25, n_mels: int = 40, n_fft: int = 1024, hop: int = 512) -> np.ndarray:
    """一打の先頭 `head_sec` 秒の対数メルスペクトログラムを 1 本のベクトルにした埋め込み。

    6 個の特徴（`segment_features`）より細かく音色を表す。計画書 3.4 の候補 (b)「埋め込みで近い音を探す」の最小版。
    長さは固定（短い音は無音で埋める）。音量の影響を減らすため、全体の最大値を 0 dB に合わせる。
    """
    from .features import FeatureConfig, mel_filterbank

    n = int(head_sec * sr)
    x = np.asarray(a, dtype=np.float64)[:n]
    x = np.pad(x, (0, max(n - x.size, 0)))
    x = x / (np.max(np.abs(x)) + 1e-9)
    cfg = FeatureConfig(sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels, f_min=30.0, f_max=min(16000.0, sr / 2 * 0.95))
    from scipy.signal import stft

    _, _, z = stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann", padded=True, boundary="zeros")
    mel = mel_filterbank(cfg) @ (np.abs(z) ** 2)
    return np.log1p(1000.0 * mel).ravel()


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    """小さな k-means（k-means++ の初期化）。返り値は各中心に最も近い点の添字（代表）。"""
    n = x.shape[0]
    k = min(k, n)
    centers = [x[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / (d2.sum() + 1e-12) if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(x[int(rng.choice(n, p=probs))])
    centers = np.asarray(centers)
    for _ in range(iters):
        assign = np.argmin([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0)
        new = np.array([x[assign == j].mean(axis=0) if np.any(assign == j) else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    reps = sorted({int(np.argmin(np.sum((x - c) ** 2, axis=1))) for c in centers})
    return np.asarray(reps)


MIN_CONFIDENCE = 0.8  # 代表音の候補にする、分離の確からしさの下限（`confident_segments`）


def attach_confidence(segments: list[Segment], hits: list[Hit], confidence: list[float | None]) -> None:
    """切り出した区間に、元の打の分離の確からしさを付ける（`hits` と `confidence` は同じ順。区間の時刻は元の打点と同じ）。"""
    by_hit = {(to_coarse(h.cls), h.time): c for h, c in zip(hits, confidence)}
    for s in segments:
        s.confidence = by_hit.get((s.cls, s.time))


def confident_segments(segs: list[Segment], min_confidence: float = MIN_CONFIDENCE) -> list[Segment]:
    """同じ種類の区間のうち、分離の確からしさの低いものを外す（代表音の候補）。

    他の音と重なった打（スネアと同時に鳴ったハイハットなど）から切り出した音は、大半が別の太鼓の音になる。StemGMD の本当の
    パート別録音で測ると、外さない場合、ハイハットの代表音の 31〜33% が「SN 比 3 dB 未満」（ほぼ別の音）で、0.8 で外すと 1〜3% に減る。
    下限は `min_confidence` と、その種類の中央値の小さい方（どの打も重なっている種類でも、確からしい方の半分は残す）。
    確からしさの分からない区間（None）は残す。
    """
    known = [s.confidence for s in segs if s.confidence is not None]
    if not known or min_confidence <= 0:
        return segs
    threshold = min(min_confidence, float(np.median(known)))
    return [s for s in segs if s.confidence is None or s.confidence >= threshold]


def build_kit_by_features(segments: list[Segment], sr: int, n_per_class: int | dict[str, int] = 12, seed: int = 0,
                          classes: tuple[str, ...] = COARSE_CLASSES, feature_fn=None, min_confidence: float = MIN_CONFIDENCE) -> Kit:
    """種類ごとに、特徴空間の k-means の代表 `n_per_class` 音でキットを作る。

    `feature_fn(audio, sr)` は既定で `segment_features`（音量・減衰・明るさ・帯域の 6 個）。`head_embedding` を渡すと
    先頭 0.25 秒の対数メルで選ぶ。3 層 × 3 音のように音量だけで選ぶのではなく、奏法の違いも代表に残る。
    再生時は各打の特徴に最も近い音を選び、音量を合わせる（`render_kit(..., hit_features=...)`。同じ feature_fn を使うこと）。
    区間に分離の確からしさが付いていれば（`attach_confidence`）、低いものは代表にしない（`confident_segments`。0 で無効）。
    """
    feature_fn = feature_fn or segment_features
    rng = np.random.default_rng(seed)
    parts: dict[str, list[Layer]] = {}
    feat_norm: dict[str, tuple[list[float], list[float]]] = {}
    for cls in classes:
        segs = confident_segments([s for s in segments if s.cls == cls], min_confidence)
        if not segs:
            continue
        feats = np.array([feature_fn(s.audio, sr) for s in segs])
        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-3
        z = (feats - mean) / std
        n_cls = n_per_class.get(cls, 12) if isinstance(n_per_class, dict) else n_per_class
        reps = _kmeans(z, n_cls, rng) if 0 < n_cls < len(segs) else np.arange(len(segs))  # n_cls <= 0 なら全部を代表にする
        layer = Layer(1, 127)
        for i in sorted(reps, key=lambda i: feats[i, 0]):
            layer.samples.append(segs[i].audio)
            layer.source_times.append(segs[i].time)
            layer.peaks_db.append(float(feats[i, 0]))
            layer.features.append([float(v) for v in feats[i]])
            layer.confidences.append(segs[i].confidence)
        parts[cls] = [layer]
        feat_norm[cls] = ([float(v) for v in mean], [float(v) for v in std])
    return Kit(sr, parts, feat_norm)


TAIL_CUT_DB = -35.0  # 区間の終わりの大きさがピークからこれより大きければ、余韻の途中で切れている（次の同じ種類の打が近かった）


def tail_level_db(a: np.ndarray, sr: int) -> float:
    """切り出した音の終わり（フェードの前の 20 ms）の大きさ ÷ ピーク（dB）。0 に近いほど、余韻の途中で切れている。"""
    n1, n0 = int(0.03 * sr), int(0.01 * sr)
    if a.size < n1 + 8:
        return 0.0
    return float(20.0 * np.log10(np.sqrt(np.mean(a[-n1:-n0] ** 2)) / (np.max(np.abs(a)) + 1e-12) + 1e-12))


def _timbre(features: np.ndarray) -> np.ndarray:
    """音量によらない特徴（`segment_features` から）: 減衰、明るさ、帯域の配分（ピークからの差）。[n, 5]。"""
    f = np.atleast_2d(np.asarray(features, dtype=float))
    return np.stack([f[:, 1], f[:, 2], f[:, 3] - f[:, 0], f[:, 4] - f[:, 0], f[:, 5] - f[:, 0]], axis=1)


def build_kit_by_level(segments: list[Segment], sr: int, n_per_class: int | dict[str, int] = 24,
                       classes: tuple[str, ...] = COARSE_CLASSES, min_confidence: float = MIN_CONFIDENCE) -> Kit:
    """MIDI だけで鳴らすためのキット（製品のキット）: 種類ごとに、音量の順に `n_per_class` 個の区分に分け、各区分から
    「普通の音」を 1 つずつ選ぶ。

    MIDI で鳴らすときは、強さ（音量）だけで音を選ぶ（`render_midi`、プラグインの `SampleEngine`）。`build_kit_by_features`
    （特徴の k-means の代表。方式 C 用）は音色のばらつきを覆うように選ぶので、他の音が混ざった打や余韻の切れた打も代表になり、
    それが音量の近い普通の打に当たる。ここでは:

    - 区分は、音量の順に同じ数ずつ（打の多い音量帯に音を多く割く）。
    - 各区分で、音色（減衰・明るさ・帯域の配分）がその区分の中央に最も近い音を選ぶ。
    - 余韻の途中で切れていない音（`tail_level_db` < `TAIL_CUT_DB`）が区分にあれば、その中から選ぶ。
    - 分離の確からしさの低い打は、先に候補から外す（`confident_segments`）。

    クラウドの標本での MIDI だけの距離（k-means の代表 → この選び方。正解の打点／取りこぼしあり）:
    StemGMD 1.041／1.537 → 1.017／1.502、E-GMD 0.917／1.178 → 0.837／1.130（2026-11-17）。
    副作用: 普通の音を選ぶので、明るい・長い打（開いたハイハットなど）の分のエネルギーが出ず、超高域（10 kHz 以上）が
    0.3〜1 dB 下がる。音量だけで選ぶ限り残る（奏法を別のノートにするまで）。
    """
    parts: dict[str, list[Layer]] = {}
    feat_norm: dict[str, tuple[list[float], list[float]]] = {}
    for cls in classes:
        segs = confident_segments([s for s in segments if s.cls == cls], min_confidence)
        if not segs:
            continue
        feats = np.array([segment_features(s.audio, sr) for s in segs])
        n_cls = n_per_class.get(cls, 24) if isinstance(n_per_class, dict) else n_per_class
        if 0 < n_cls < len(segs):
            timbre = _timbre(feats)
            z = (timbre - np.median(timbre, axis=0)) / (timbre.std(axis=0) + 1e-3)
            whole = np.array([tail_level_db(s.audio, sr) < TAIL_CUT_DB for s in segs])
            chosen = []
            for group in np.array_split(np.argsort(feats[:, 0], kind="stable"), n_cls):
                cand = group[whole[group]] if np.any(whole[group]) else group
                center = np.median(z[group], axis=0)
                chosen.append(int(cand[int(np.argmin(((z[cand] - center) ** 2).sum(axis=1)))]))
        else:  # n_cls <= 0 か、打が少なければ全部
            chosen = list(range(len(segs)))
        layer = Layer(1, 127)
        for i in sorted(chosen, key=lambda i: feats[i, 0]):
            layer.samples.append(segs[i].audio)
            layer.source_times.append(segs[i].time)
            layer.peaks_db.append(float(feats[i, 0]))
            layer.features.append([float(v) for v in feats[i]])
            layer.confidences.append(segs[i].confidence)
        parts[cls] = [layer]
        feat_norm[cls] = ([float(v) for v in feats.mean(axis=0)], [float(v) for v in feats.std(axis=0) + 1e-3])
    return Kit(sr, parts, feat_norm)


def hit_segments_from_parts(parts: dict[str, tuple[np.ndarray, int]], hits: list[Hit],
                            max_len_sec: float | dict[str, float] | None = None) -> list[np.ndarray | None]:
    """各打の「自分の区間」（打点から同じ種類の次の打まで、上限は種類ごと）をパート別ステムから取る。hits と同じ順。"""
    out: list[np.ndarray | None] = [None] * len(hits)
    by_cls: dict[str, list[int]] = {}
    for i, h in enumerate(hits):
        by_cls.setdefault(to_coarse(h.cls), []).append(i)
    for cls, idx in by_cls.items():
        if cls not in parts:
            continue
        x, sr = parts[cls]
        idx = sorted(idx, key=lambda i: hits[i].time)
        limit = max_len_for(cls, max_len_sec)
        for k, i in enumerate(idx):
            t = hits[i].time
            next_t = hits[idx[k + 1]].time if k + 1 < len(idx) else t + limit
            start = int(round(t * sr))
            end = min(start + int(min(next_t - t, limit) * sr), x.size)
            if end - start >= int(0.005 * sr):
                out[i] = x[start:end]
    return out


def build_kit(
    segments: list[Segment],
    sr: int,
    n_layers: int = 3,
    round_robin: int = 3,
    classes: tuple[str, ...] = COARSE_CLASSES,
) -> Kit:
    """種類ごとにピークの大きさで層に分ける。打が少ない種類は層を減らす。"""
    parts: dict[str, list[Layer]] = {}
    for cls in classes:
        segs = sorted((s for s in segments if s.cls == cls), key=lambda s: s.peak_db)
        if not segs:
            continue
        k = max(1, min(n_layers, len(segs)))
        # ピークの分位で層の境界を決める
        bounds = np.linspace(0, len(segs), k + 1).astype(int)
        layers: list[Layer] = []
        for li in range(k):
            chunk = segs[bounds[li] : bounds[li + 1]]
            if not chunk:
                continue
            vel_lo = 1 + int(round(126 * li / k))
            vel_hi = int(round(127 * (li + 1) / k))
            # 層の中では代表的な（中央寄りの）打からラウンドロビンに使う
            mid = len(chunk) // 2
            order = sorted(range(len(chunk)), key=lambda j: abs(j - mid))[:round_robin]
            layer = Layer(vel_lo, vel_hi)
            for j in sorted(order):
                layer.samples.append(chunk[j].audio)
                layer.source_times.append(chunk[j].time)
                layer.peaks_db.append(chunk[j].peak_db)
            layers.append(layer)
        parts[cls] = layers
    return Kit(sr, parts)


def _onset_lag(target: np.ndarray, sample: np.ndarray, sr: int, max_lag_sec: float = 0.015, margin: float = 1.0) -> int:
    """`sample` を何サンプルずらせば立ち上がりが `target` に重なるか（相互相関、±max_lag_sec）。正なら遅らせる。

    比べるのは、頭から 40 ms の波形の絶対値。`margin` が 1 より大きければ、最良の位置の相関が「ずらさない位置」の相関の `margin` 倍
    以上のときだけずらす（それ未満なら 0）。低い音の太鼓は、周期ぶん滑った位置でも相関がほぼ同じになることがある。
    """
    n = min(int(0.04 * sr), target.size, sample.size)
    if n < 16:
        return 0
    a = np.abs(target[:n]).astype(np.float64)
    b = np.abs(sample[:n]).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    max_lag = min(int(max_lag_sec * sr), n // 2)  # 短い音では、ずらす幅を比較する長さの半分までに抑える
    if max_lag <= 0:
        return 0
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(a[lag:], b[: n - lag]))
        else:
            c = float(np.dot(a[: n + lag], b[-lag:]))
        if c > best:
            best, best_lag = c, lag
    if margin > 1.0 and best_lag != 0:
        c0 = float(np.dot(a, b))
        if c0 > 0 and best < margin * c0:
            return 0
    return best_lag


def render_kit(
    hits: list[Hit],
    kit: Kit,
    duration: float | None = None,
    hit_levels_db: list[float | None] | None = None,
    hit_features: list[np.ndarray | None] | None = None,
    gain_match: bool = False,
    choke: bool | set[str] = False,
    max_gain_db: float = 12.0,
    hit_audio: list[np.ndarray | None] | None = None,
    align: bool = False,
    leave_one_out: bool = False,
) -> np.ndarray:
    """打点ラベルをキットで鳴らす。

    - 音の選び方: `hit_features` があれば（特徴の入ったキットで）最も近い音。無ければ強さの層とラウンドロビン。
      強さが無い打は最も強い層を使う。
    - `gain_match`: `hit_levels_db`（その打の元のピーク dB）に合わせて音量を調整する（±max_gain_db まで）。
    - `choke`: 同じ種類の次の打が来たら前の音を 5 ms で止める。True なら `CHOKE_CLASSES`（シンバル以外）、
      種類の集合（例: {"hihat"}）ならその種類だけ、False なら止めない。
    - `align`: `hit_audio`（その打自身の区間）があれば、選んだ音の立ち上がりを相互相関で元に合わせて置く（±15 ms）。
      打点の推定は約 10 ms 刻みなので、そのままだと最大 1 フレームずれる。
    - `leave_one_out`: 選ぶときに、その打自身に由来する音を候補から外す（全部を代表にしたときの上限の測定用）。
    `hit_levels_db`／`hit_features`／`hit_audio` は `hits` と同じ順の並び。
    """
    sr = kit.sr
    last = max((h.time for h in hits), default=0.0)
    if duration is None:
        duration = last + 1.5
    n = int(duration * sr)
    out = np.zeros(n)
    rr_index: dict[tuple[str, int], int] = {}
    order = sorted(range(len(hits)), key=lambda i: hits[i].time)
    # 同じ種類の次の打の時刻（チョーク用）
    next_same: dict[int, float] = {}
    last_of_cls: dict[str, int] = {}
    for i in order:
        c = to_coarse(hits[i].cls)
        if c in last_of_cls:
            next_same[last_of_cls[c]] = hits[i].time
        last_of_cls[c] = i
    fade = max(int(0.005 * sr), 1)
    choke_set: set[str] = CHOKE_CLASSES if choke is True else (choke if isinstance(choke, set) else set())

    for i in order:
        h = hits[i]
        cls = to_coarse(h.cls)
        layers = kit.parts.get(cls)
        if not layers:
            continue
        chosen = None
        if hit_features is not None and hit_features[i] is not None:
            chosen = kit.pick_nearest(cls, hit_features[i], exclude_time=h.time if leave_one_out else None)
        if chosen is None:
            if h.velocity is None:
                li = len(layers) - 1
            else:
                li = next((k for k, l in enumerate(layers) if l.vel_lo <= h.velocity <= l.vel_hi), len(layers) - 1)
            layer = layers[li]
            key = (cls, li)
            j = rr_index.get(key, 0) % len(layer.samples)
            rr_index[key] = j + 1
            peak = layer.peaks_db[j] if j < len(layer.peaks_db) else 20.0 * np.log10(np.max(np.abs(layer.samples[j])) + 1e-9)
            chosen = (layer.samples[j], peak)
        s, peak_db = chosen
        g = 1.0
        if gain_match and hit_levels_db is not None and hit_levels_db[i] is not None:
            g = 10 ** (float(np.clip(hit_levels_db[i] - peak_db, -max_gain_db, max_gain_db)) / 20)
        start = int(round(h.time * sr))
        if align and hit_audio is not None and hit_audio[i] is not None:
            start += _onset_lag(hit_audio[i], s, sr)
        if start >= n or start < 0:
            continue
        length = min(s.size, n - start)
        if cls in choke_set and i in next_same:
            limit = int((next_same[i] - h.time) * sr) + fade
            length = min(length, max(limit, fade))
        seg = s[:length] * g
        if length < s.size:
            seg = seg.copy()
            k = min(fade, seg.size)
            seg[-k:] *= np.linspace(1.0, 0.0, k)
        out[start : start + length] += seg
    return out


CHOKE_FADE_SEC = 0.005  # チョーク（同じ種類の次の打で前の音を止める）のフェードの長さ
MAX_BOOST_DB = 12.0  # MIDI で鳴らすとき、選んだ音を持ち上げる上限（小さい音を大きく鳴らすと、雑音やにじみも持ち上がる）。下げる側は制限しない


def render_midi(hits: list[Hit], kit: Kit, duration: float | None = None, choke: bool | set[str] = True, neighbours: int = 1,
                max_boost_db: float = MAX_BOOST_DB, window_db: float | None = None) -> np.ndarray:
    """MIDI の情報（時刻・種類・強さ）だけでキットを鳴らす。プラグインの `SampleEngine` と同じ鳴らし方（参照実装）。

    プラグイン（`plugin/Source/SampleEngine.cpp`）と 1 サンプル単位で同じ結果になるように書いてある（`engine_parity.py` と CI で確かめる）。
    鳴らし方を変えるときは両方を変える。

    - 強さ → 目標のピーク（`kit.level_range` の範囲で dB に線形。`level_for_velocity`）。
    - その種類の音のうち、ピークが目標に最も近い音を使う（同じ近さなら、ピークの小さい方）。`neighbours` を 2 以上にすると、
      近い順にその数の音を順に使う（同じ強さが続いても同じ音ばかりにならないが、元の再現からは遠ざかる。E-GMD と StemGMD の標本で、
      ±0.5 dB の幅でも距離が 0.12〜0.17 増えた。音量が近くても奏法の違う音が混ざるため。プラグインは 1）。`window_db` はその幅の上限。
    - 音量を目標に合わせる。持ち上げるのは `max_boost_db` まで、下げる側は制限しない。
    - 音は打点の `SAMPLE_LEAD_SEC` 前から置く（切り出した音の立ち上がりが打点に重なる。プラグインは打点から鳴らし始め、
      この長さを遅れとしてホストに伝える）。
    - `choke`: 同じ種類の次の打の立ち上がりから、前の音を `CHOKE_FADE_SEC` で止める（True ならシンバル以外）。
    `render_kit(..., hit_features=...)`（方式 C）は元の音の特徴で打ごとに音を選ぶので、MIDI には無い情報を使っている。
    """
    sr = kit.sr
    last = max((h.time for h in hits), default=0.0)
    if duration is None:
        duration = last + 1.5
    n = int(duration * sr)
    out = np.zeros(n)
    order = sorted(range(len(hits)), key=lambda i: hits[i].time)
    pos = {i: int(round(hits[i].time * sr)) for i in order}  # 打点のサンプル位置（プラグインには、この位置でノートオンが来る）
    next_same: dict[int, int] = {}
    last_of_cls: dict[str, int] = {}
    for i in order:
        c = to_coarse(hits[i].cls)
        if c in last_of_cls:
            next_same[last_of_cls[c]] = pos[i]
        last_of_cls[c] = i
    fade = max(int(CHOKE_FADE_SEC * sr), 1)
    ramp = 1.0 - np.arange(fade) / fade  # 1 から始まり、fade サンプル目で 0（そこで音を止める）
    choke_set: set[str] = CHOKE_CLASSES if choke is True else (choke if isinstance(choke, set) else set())
    lead = int(round(SAMPLE_LEAD_SEC * sr))
    rr: dict[str, int] = {}
    pools: dict[str, list[tuple[np.ndarray, float]]] = {}
    for cls in kit.parts:
        pools[cls] = sorted(((smp, peak) for smp, peak, _ in kit.all_samples(cls)), key=lambda t: t[1])
    for i in order:
        h = hits[i]
        cls = to_coarse(h.cls)
        pool = pools.get(cls)
        if not pool:
            continue
        peaks = np.array([p for _, p in pool])
        rng_db = kit.level_range.get(cls) or level_range_for(peaks)
        target = level_for_velocity(h.velocity, rng_db)
        near = np.argsort(np.abs(peaks - target), kind="stable")[: max(min(neighbours, len(pool)), 1)]
        if window_db is not None:  # 最も近い音から window_db 以内の音だけを順に使う
            dist = np.abs(peaks[near] - target)
            near = near[dist <= dist[0] + window_db]
        j = int(near[rr.get(cls, 0) % near.size])
        rr[cls] = rr.get(cls, 0) + 1
        s, peak_db = pool[j]
        seg = s * 10 ** (min(target - peak_db, max_boost_db) / 20)
        if cls in choke_set and i in next_same:
            stop = next_same[i] - pos[i] + lead  # 音の頭から数えた、止め始める位置（次の打の立ち上がり）
            if stop < seg.size:
                k = min(fade, seg.size - stop)
                seg[stop : stop + k] *= ramp[:k]
                seg = seg[: stop + k]
        start = pos[i] - lead
        skip = max(-start, 0)
        start = max(start, 0)
        length = min(seg.size - skip, n - start)
        if length > 0:
            out[start : start + length] += seg[skip : skip + length]
    return out


def estimate_velocities(hits: list[Hit], segments_or_peaks: dict[str, list[tuple[float, float]]] | None, x: np.ndarray, sr: int) -> list[Hit]:
    """打点ごとの立ち上がりの大きさ（打点から 20 ms のピーク）を、種類ごとの分布で 1〜127 に割り当てる（計画書 3.2）。

    最小のピーク → 30、最大 → 127 の線形。打が1つしか無い種類は 100。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    win = int(0.02 * sr)
    peaks = []
    for h in hits:
        s = int(round(h.time * sr))
        seg = x[max(s, 0) : s + win]
        peaks.append(20.0 * np.log10(np.max(np.abs(seg)) + 1e-9) if seg.size else -100.0)
    peaks_arr = np.asarray(peaks)
    out: list[Hit] = []
    for cls in {to_coarse(h.cls) for h in hits}:
        idx = [i for i, h in enumerate(hits) if to_coarse(h.cls) == cls]
        lo, hi = peaks_arr[idx].min(), peaks_arr[idx].max()
        for i in idx:
            if hi - lo < 1e-6:
                v = 100
            else:
                v = int(round(30 + (peaks_arr[i] - lo) / (hi - lo) * 97))
            out.append(Hit(hits[i].time, hits[i].cls, max(1, min(127, v))))
    out.sort(key=lambda h: h.time)
    return out


def save_kit(kit: Kit, out_dir: str | Path) -> None:
    import soundfile as sf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"sr": kit.sr, "parts": {}}
    for cls, layers in kit.parts.items():
        (out / cls).mkdir(exist_ok=True)
        meta["parts"][cls] = []
        for li, layer in enumerate(layers):
            files = []
            for j, s in enumerate(layer.samples):
                name = f"{cls}/layer{li + 1}_rr{j + 1}.wav"
                sf.write(out / name, s.astype(np.float32), kit.sr, subtype="PCM_24")  # 小さい音を持ち上げても粗くならないように 24 bit
                files.append(name)
            meta["parts"][cls].append({"vel_lo": layer.vel_lo, "vel_hi": layer.vel_hi, "files": files, "source_times": layer.source_times,
                                       "peaks_db": layer.peaks_db, "features": layer.features, "confidence": layer.confidences})
    meta["feat_norm"] = kit.feat_norm
    meta["level_range_db"] = {c: [float(lo), float(hi)] for c, (lo, hi) in kit.level_range.items()}
    meta["lead_sec"] = SAMPLE_LEAD_SEC  # 各音は打点のこの秒数だけ前から始まる（立ち上がりは音の頭からこの位置）
    (out / "kit.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_kit(kit_dir: str | Path) -> Kit:
    import soundfile as sf

    d = Path(kit_dir)
    meta = json.loads((d / "kit.json").read_text(encoding="utf-8"))
    parts: dict[str, list[Layer]] = {}
    for cls, layers in meta["parts"].items():
        parts[cls] = []
        for l in layers:
            layer = Layer(l["vel_lo"], l["vel_hi"], source_times=list(l.get("source_times", [])),
                          peaks_db=list(l.get("peaks_db", [])), features=list(l.get("features", [])),
                          confidences=list(l.get("confidence", [])))
            for f in l["files"]:
                a, _ = sf.read(d / f, dtype="float64", always_2d=True)
                layer.samples.append(a.mean(axis=1))
            parts[cls].append(layer)
    feat_norm = {k: (list(v[0]), list(v[1])) for k, v in meta.get("feat_norm", {}).items()}
    level_range = {k: (float(v[0]), float(v[1])) for k, v in meta.get("level_range_db", {}).items()}
    return Kit(int(meta["sr"]), parts, feat_norm, level_range)
