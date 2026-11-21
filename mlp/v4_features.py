"""V4 wave-chain feature builder for the production MLP server.

This module is the single source of truth for the 38-key V4 state sent to
``/v1/systemone``. It mirrors the GUI path in ``wnf-elwave-test`` and the
training-data basis in ``generate_v4_dataset.py``:

* Neely 1-D series with the caller's ``pivot_type`` (0/1/2, same choices as
  the GUI). Default 0 is the canonical training/model basis; 1 and 2 are the
  Williams-fractal variants and produce a different monowave list, hence a
  different wave chain and V4 state.
* ``MacroWave.monoize()`` + ``analyze(EPAM.SI)`` as in ``wave_analysis.py``.
* Wave chain ``w_adj_ws(s, e, 5)`` / ``w_adj_ws(s, e, -2)`` as in
  ``wave_chain_features.py`` / ``generate_v4_dataset.py``.
* ``Decimal(..., ROUND_HALF_UP)`` rounding as in the training dataset.

The vendored modules live in ``mlp/vendor/`` so this server has no runtime
dependency on the GUI checkout.
"""

from __future__ import annotations

import os
import signal
import sys
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from neely import generate_neely_wave_linear  # noqa: E402
from wnf_elliottwave import EPAM, MacroWave  # noqa: E402


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover - main thread only
    raise _TimeoutError("wave chain timeout")


def _with_timeout(func, timeout: float = 2.0):
    """Run ``func`` with an alarm when in the main thread.

    ``signal.alarm`` is only available in the main thread; the HTTP server
    runs handlers in worker threads, so fall back to a plain call there.
    """
    import threading

    if threading.current_thread() is not threading.main_thread():
        return func(), False

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout))
    try:
        result = func()
        signal.alarm(0)
        return result, False
    except _TimeoutError:
        signal.alarm(0)
        return None, True
    except Exception:
        signal.alarm(0)
        raise


def round3(v: float | None):
    """Training-data rounding: Decimal quantize with ROUND_HALF_UP."""
    if v is None:
        return None
    q = float(Decimal(v).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return int(q) if q.is_integer() else q


def clip_retrace(v: float | None) -> float | None:
    if v is None:
        return None
    return max(0.1, min(10.0, v))


def wave_chain(mw: MacroWave, s: int, e: int, timeout: float = 2.0):
    """Return (forward, backward) endpoint chains for wave [s, e]."""
    try:
        fwd, _ = _with_timeout(lambda: mw.w_adj_ws(s, e, 5), timeout)
    except Exception:
        fwd = []
    try:
        bwd, _ = _with_timeout(lambda: mw.w_adj_ws(s, e, -2), timeout)
    except Exception:
        bwd = []
    return fwd or [], bwd or []


def chain_waves(fwd, bwd, s: int, e: int) -> dict[str, tuple[int, int]]:
    """Map V4 wave names to their (start, end) bar indices."""
    waves: dict[str, tuple[int, int]] = {"m0": (s, e)}
    if len(fwd) >= 3:
        waves["m1"] = (fwd[1], fwd[2])
    if len(fwd) >= 4:
        waves["m2"] = (fwd[2], fwd[3])
    if len(fwd) >= 5:
        waves["m3"] = (fwd[3], fwd[4])
    if len(fwd) >= 6:
        waves["m4"] = (fwd[4], fwd[5])
    if len(bwd) >= 1:
        waves["m_1"] = (bwd[0], s)
    if len(bwd) >= 2:
        waves["m_2"] = (bwd[1], bwd[0])
    if len(bwd) >= 3:
        waves["m_3"] = (bwd[2], bwd[1])
    return waves


def build_v4_state(mw: MacroWave, s: int, e: int, ohlc, prices) -> dict[str, Any]:
    """Build the 38-key V4 state for wave [s, e] (training-data basis)."""
    fwd, bwd = wave_chain(mw, s, e)
    waves = chain_waves(fwd, bwd, s, e)

    def wave_info(name: str):
        if name not in waves:
            return None
        ws, we = waves[name]
        seg = ohlc[ws : we + 1]
        if not seg:
            return None
        start_price = prices[ws]
        end_price = prices[we]
        return {
            "dir": "up" if end_price > start_price else "down",
            "length": abs(end_price - start_price),
            "timeBars": we - ws + 1,
            "high": max(d["high"] for d in seg),
            "low": min(d["low"] for d in seg),
        }

    m0 = wave_info("m0")
    m1 = wave_info("m1")
    m2 = wave_info("m2")
    m3 = wave_info("m3")
    m4 = wave_info("m4")
    m_1 = wave_info("m_1")
    m_2 = wave_info("m_2")
    m_3 = wave_info("m_3")

    if m0 is None:
        raise ValueError(f"invalid target wave span [{s}, {e}]")

    m0_dir = m0["dir"]
    m0_length = m0["length"]
    m0_change_pct = ((prices[e] - prices[s]) / prices[s]) * 100 if prices[s] != 0 else 0
    m0_time_bars = m0["timeBars"]

    # Engine RC decision ratios (signed, unclipped).
    lc = prices[e] - prices[s]
    rc_la_lc = None
    rc_lb_lc = None
    if lc != 0 and len(fwd) >= 3:
        la = prices[fwd[2]] - prices[fwd[1]]
        rc_la_lc = la / lc
    if lc != 0 and len(bwd) >= 1:
        lb = prices[s] - prices[bwd[0]]
        rc_lb_lc = lb / lc

    m1_retrace_ratio = clip_retrace(
        (m1["length"] / m0_length) if (m1 and m0_length != 0) else None
    )
    m2_retrace_ratio = clip_retrace(
        (m2["length"] / m1["length"]) if (m2 and m1 and m1["length"] != 0) else None
    )
    m_1_retrace_ratio = clip_retrace(
        (m_1["length"] / m0_length) if (m_1 and m0_length != 0) else None
    )
    m_2_retrace_ratio = clip_retrace(
        (m_2["length"] / m1["length"]) if (m_2 and m1 and m1["length"] != 0) else None
    )
    m3_retrace_ratio = clip_retrace(
        (m3["length"] / m2["length"]) if (m3 and m2 and m2["length"] != 0) else None
    )
    m4_retrace_ratio = clip_retrace(
        (m4["length"] / m3["length"]) if (m4 and m3 and m3["length"] != 0) else None
    )

    m1_time_ratio = (m1["timeBars"] / m0_time_bars) if (m1 and m0_time_bars != 0) else None
    m2_time_ratio = (m2["timeBars"] / m1["timeBars"]) if (m2 and m1 and m1["timeBars"] != 0) else None
    m_1_time_ratio = (m_1["timeBars"] / m0_time_bars) if (m_1 and m0_time_bars != 0) else None
    m_2_time_ratio = (m_2["timeBars"] / m1["timeBars"]) if (m_2 and m1 and m1["timeBars"] != 0) else None
    m3_time_ratio = (m3["timeBars"] / m2["timeBars"]) if (m3 and m2 and m2["timeBars"] != 0) else None
    m4_time_ratio = (m4["timeBars"] / m3["timeBars"]) if (m4 and m3 and m3["timeBars"] != 0) else None

    m1_time_bars = m1["timeBars"] if m1 else None
    m2_time_bars = m2["timeBars"] if m2 else None
    m_1_time_bars = m_1["timeBars"] if m_1 else None
    m_2_time_bars = m_2["timeBars"] if m_2 else None
    m3_time_bars = m3["timeBars"] if m3 else None
    m4_time_bars = m4["timeBars"] if m4 else None
    m_3_time_bars = m_3["timeBars"] if m_3 else None
    m_4_time_bars = None

    def overlap(w_a, w_b) -> bool:
        if w_a is None or w_b is None:
            return False
        return w_a["low"] < w_b["high"] and w_a["high"] > w_b["low"]

    seg0 = ohlc[s : e + 1]
    m0_high = max(d["high"] for d in seg0)
    m0_low = min(d["low"] for d in seg0)

    m_2_vs_m0_overlap = overlap(m_2, {"high": m0_high, "low": m0_low}) if m_2 else False
    m_1_vs_m1_overlap = overlap(m_1, m1) if (m_1 and m1) else False
    m0_vs_m2_overlap = overlap({"high": m0_high, "low": m0_low}, m2) if m2 else False
    m1_vs_m3_overlap = overlap(m1, m3) if (m1 and m3) else False
    m2_vs_m4_overlap = overlap(m2, m4) if (m2 and m4) else False
    m_2_vs_m1_overlap = overlap(m_2, m1) if (m_2 and m1) else False
    m_1_vs_m2_overlap = overlap(m_1, m2) if (m_1 and m2) else False
    m0_vs_m3_overlap = overlap({"high": m0_high, "low": m0_low}, m3) if m3 else False
    m1_vs_m4_overlap = overlap(m1, m4) if (m1 and m4) else False

    beyond = (
        (m0_high > max(m1["high"], m2["high"]) if (m1 and m2) else False)
        or (m0_low < min(m1["low"], m2["low"]) if (m1 and m2) else False)
    )

    m_1_mw_count = mw.w_mw_cnt(waves["m_1"][0], waves["m_1"][1]) if "m_1" in waves else 0
    m2_mw_count = mw.w_mw_cnt(waves["m2"][0], waves["m2"][1]) if "m2" in waves else 0
    m3_mw_count = mw.w_mw_cnt(waves["m3"][0], waves["m3"][1]) if "m3" in waves else 0

    return {
        "m0_dir": m0_dir,
        "m0_change_pct": round3(m0_change_pct),
        "rc_la_lc": round3(rc_la_lc),
        "rc_lb_lc": round3(rc_lb_lc),
        "m1_retrace_ratio": round3(m1_retrace_ratio),
        "m2_retrace_ratio": round3(m2_retrace_ratio),
        "m_1_retrace_ratio": round3(m_1_retrace_ratio),
        "m_2_retrace_ratio": round3(m_2_retrace_ratio),
        "m3_retrace_ratio": round3(m3_retrace_ratio),
        "m4_retrace_ratio": round3(m4_retrace_ratio),
        "m1_time_ratio": round3(m1_time_ratio),
        "m2_time_ratio": round3(m2_time_ratio),
        "m_1_time_ratio": round3(m_1_time_ratio),
        "m_2_time_ratio": round3(m_2_time_ratio),
        "m3_time_ratio": round3(m3_time_ratio),
        "m4_time_ratio": round3(m4_time_ratio),
        "m0_time_bars": m0_time_bars,
        "m1_time_bars": m1_time_bars,
        "m2_time_bars": m2_time_bars,
        "m_1_time_bars": m_1_time_bars,
        "m_2_time_bars": m_2_time_bars,
        "m3_time_bars": m3_time_bars,
        "m4_time_bars": m4_time_bars,
        "m_3_time_bars": m_3_time_bars,
        "m_4_time_bars": m_4_time_bars,
        "m_2_vs_m0_overlap": m_2_vs_m0_overlap,
        "m_1_vs_m1_overlap": m_1_vs_m1_overlap,
        "m0_vs_m2_overlap": m0_vs_m2_overlap,
        "m1_vs_m3_overlap": m1_vs_m3_overlap,
        "m2_vs_m4_overlap": m2_vs_m4_overlap,
        "m_2_vs_m1_overlap": m_2_vs_m1_overlap,
        "m_1_vs_m2_overlap": m_1_vs_m2_overlap,
        "m0_vs_m3_overlap": m0_vs_m3_overlap,
        "m1_vs_m4_overlap": m1_vs_m4_overlap,
        "beyond": beyond,
        "m_1_mw_count": m_1_mw_count,
        "m2_mw_count": m2_mw_count,
        "m3_mw_count": m3_mw_count,
    }


def _first_value(bar: dict, names: tuple[str, ...]) -> float:
    for name in names:
        if name in bar and bar[name] is not None and bar[name] != "":
            return float(bar[name])
    raise KeyError(f"OHLC bar is missing any of {names}: {bar}")


def normalize_candles(candles: list[dict]) -> list[dict]:
    """Accept both chart-proxy and GUI OHLC key names."""
    out: list[dict] = []
    for bar in candles:
        high = _first_value(bar, ("high", "high_pric", "High"))
        low = _first_value(bar, ("low", "low_pric", "Low"))
        close = _first_value(bar, ("close", "cur_prc", "Close", "close_pric"))
        out.append(
            {
                "high_pric": abs(high),
                "low_pric": abs(low),
                "cur_prc": abs(close),
                # build_v4_state reads high/low directly.
                "high": abs(high),
                "low": abs(low),
                "close": abs(close),
            }
        )
    return out


def build_features_response(
    candles: list[dict],
    pivot_type: int = 0,
    mode: str = "SI",
) -> dict[str, Any]:
    """Build monowaves and V4 states from raw OHLC candles.

    ``pivot_type`` mirrors the GUI combo box: 0=Neely (training basis),
    1=Williams Fractal N=2, 2=Williams N=2 + Alligator. Changing it changes
    the monowave list, the ``w_adj_ws`` chain and therefore every V4 state.
    ``mode`` (SI/DI/DE) is the GUI's "Analysis mode" and selects the EPAM used
    by ``MacroWave.analyze()``.

    Returns a payload consumed by ``stock-twin-kr``::

        {
          "prices": [...],
          "waves": [{"start", "end", "sp", "startPrice", "endPrice", "state"}, ...],
          "count": n
        }
    """
    if pivot_type not in (0, 1, 2):
        raise ValueError(f"unsupported pivot_type: {pivot_type!r} (expected 0, 1 or 2)")
    if mode not in ("SI", "DI", "DE"):
        raise ValueError(f"unsupported analysis mode: {mode!r}")

    ohlc = normalize_candles(candles)
    if len(ohlc) < 2:
        raise ValueError("feature builder needs at least 2 OHLC candles")

    prices = generate_neely_wave_linear(ohlc, pivot_type=pivot_type)
    macro = MacroWave()
    macro.plist = [float(p) for p in prices]
    macro.monoize()
    if not macro.mw_list:
        raise ValueError("no monowaves found in this price series")
    macro.analyze(EPAM[mode])

    waves = []
    for wave in sorted(macro.mw_list.values(), key=lambda item: item.liS):
        s, e = int(wave.liS), int(wave.liE)
        if not (0 <= s < len(prices) and 0 <= e < len(prices)):
            continue
        waves.append(
            {
                "start": s,
                "end": e,
                "sp": wave.mSP or "",
                "startPrice": float(prices[s]),
                "endPrice": float(prices[e]),
                "state": build_v4_state(macro, s, e, ohlc, prices),
            }
        )

    return {"prices": [float(p) for p in prices], "waves": waves, "count": len(waves)}
