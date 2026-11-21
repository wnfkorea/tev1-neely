"""Neely-style 1-D reconstruction of OHLC price data.

This module contains the standalone function required by the test program:
:func:`generate_neely_wave_linear` converts a list of OHLC candles into the
one-dimensional price series that is fed to ``wnf_elliottwave.MacroWave``.

It is a dependency-free port of
``puppeteer-rest-api/kiwoom_api.py::generate_neely_wave_linear``.

Supported pivot constructions
-----------------------------
0. Neely: a wave ends when the current bar breaks the previous bar's low
   (downtrend) or high (uptrend). This is the default because the test
   program is meant to follow Neely's construction.
1. Williams Fractal (N=2) with an alternating high/low filter.
2. Williams Fractal (N=2) plus the Alligator SMA(13/8/5) trend filter.

The input candles are expected in **chronological order** (oldest first) by
default. Pass ``newest_first=True`` when the data comes in the newest-first
order used by the REST endpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

__all__ = ["generate_neely_wave_linear"]

Bar = Mapping[str, float | int | str]
Pivot = tuple[int, float]


def _value(bar: Bar, key: str) -> float:
    """Read one absolute OHLC value, mirroring the original ``.abs()``."""
    try:
        return abs(float(bar[key]))
    except KeyError as exc:
        raise KeyError(f"OHLC bar is missing {key!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OHLC bar has a non-numeric {key!r}: {bar[key]!r}") from exc


def _interpolate(count: int, pivots: Sequence[Pivot]) -> list[float]:
    """Fill every bar between two pivots with a straight line."""
    values = [0.0] * count
    for (i1, p1), (i2, p2) in zip(pivots, pivots[1:]):
        span = i2 - i1
        if span <= 0:
            if 0 <= i1 < count:
                values[i1] = p1
            continue
        step = (p2 - p1) / span
        for offset in range(span + 1):
            values[i1 + offset] = p1 + step * offset
    return values


def _pivots_neely(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[Pivot]:
    """Neely pivots: reverse versus the previous bar (pivot_type=0)."""
    count = len(closes)
    pivots: list[Pivot] = [(0, closes[0])]
    trend = 1 if closes[1] > closes[0] else -1

    wave_high = highs[0]
    wave_low = lows[0]

    for i in range(1, count):
        prev_high, prev_low = highs[i - 1], lows[i - 1]
        curr_high, curr_low = highs[i], lows[i]

        is_inside = curr_high <= prev_high and curr_low >= prev_low
        reversed_trend = (
            (trend == 1 and curr_low < prev_low)
            or (trend == -1 and curr_high > prev_high)
        )
        if not is_inside and reversed_trend:
            # The end of an up wave is its high; the end of a down wave is its low.
            pivot_price = wave_high if trend == 1 else wave_low
            pivots.append((i - 1, pivot_price))
            trend *= -1
            wave_high, wave_low = curr_high, curr_low

        wave_high = max(wave_high, curr_high)
        wave_low = min(wave_low, curr_low)

    pivots.append((count - 1, closes[-1]))
    return pivots


def _pivots_williams(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    use_alligator: bool,
) -> list[Pivot]:
    """Williams Fractal(N=2) pivots, optionally filtered by the Alligator."""
    count = len(closes)
    n_bar = 2

    fractals: list[tuple[int, float, str]] = []
    for i in range(n_bar, count - n_bar):
        window_high = list(highs[i - n_bar : i + n_bar + 1])
        window_low = list(lows[i - n_bar : i + n_bar + 1])
        if highs[i] == max(window_high) and window_high.count(highs[i]) == 1:
            fractals.append((i, float(highs[i]), "H"))
        if lows[i] == min(window_low) and window_low.count(lows[i]) == 1:
            fractals.append((i, float(lows[i]), "L"))
    fractals.sort(key=lambda item: item[0])

    trend = [0] * count
    if use_alligator:
        for i in range(13, count):
            jaws = sum(closes[i - 13 : i]) / 13
            teeth = sum(closes[i - 8 : i]) / 8
            lips = sum(closes[i - 5 : i]) / 5
            if jaws > teeth > lips:
                trend[i] = 1
            elif jaws < teeth < lips:
                trend[i] = -1

    filtered: list[tuple[int, float, str]] = []
    for idx, price, direction in fractals:
        if not filtered:
            filtered.append((idx, price, direction))
            continue

        _, last_price, last_dir = filtered[-1]
        if direction == last_dir:
            # Alternating filter: keep the more extreme price of the same direction.
            if (direction == "H" and price > last_price) or (
                direction == "L" and price < last_price
            ):
                filtered[-1] = (idx, price, direction)
            continue

        if use_alligator:
            if trend[idx] != 0 and direction == "H" and trend[idx] == -1:
                continue
            if trend[idx] != 0 and direction == "L" and trend[idx] == 1:
                continue

        filtered.append((idx, price, direction))

    pivots = (
        [(0, float(closes[0]))]
        + [(idx, price) for idx, price, _ in filtered]
        + [(count - 1, float(closes[-1]))]
    )

    # Remove duplicated indices where the start/end overlaps a fractal.
    seen: set[int] = set()
    unique: list[Pivot] = []
    for idx, price in pivots:
        if idx not in seen:
            seen.add(idx)
            unique.append((idx, price))
    return unique


def generate_neely_wave_linear(
    ohlc: Iterable[Bar],
    pivot_type: int = 0,
    *,
    newest_first: bool = False,
) -> list[float]:
    """Convert OHLC candles into a 1-D price series for wave analysis.

    Parameters
    ----------
    ohlc:
        Iterable of mappings that contain ``high_pric``, ``low_pric`` and
        ``cur_prc`` (close). These are the same key names the
        ``/coin_chart`` endpoint returns.
    pivot_type:
        ``0`` = Neely construction (default), ``1`` = Williams Fractal(N=2),
        ``2`` = Williams Fractal(N=2) + Alligator trend filter.
    newest_first:
        When true, the input iterable is reversed before construction.

    Returns
    -------
    list[float]
        One reconstructed price per input bar, in chronological order.
    """
    bars = list(ohlc)
    if newest_first:
        bars.reverse()
    if not bars:
        return []
    if pivot_type not in (0, 1, 2):
        raise ValueError(f"unsupported pivot_type: {pivot_type!r} (expected 0, 1 or 2)")

    highs = [_value(bar, "high_pric") for bar in bars]
    lows = [_value(bar, "low_pric") for bar in bars]
    closes = [_value(bar, "cur_prc") for bar in bars]

    if len(bars) == 1:
        return [closes[0]]

    if pivot_type == 0:
        pivots = _pivots_neely(highs, lows, closes)
    else:
        pivots = _pivots_williams(highs, lows, closes, use_alligator=(pivot_type == 2))

    return _interpolate(len(bars), pivots)
