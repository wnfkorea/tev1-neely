"""Python port of the WnFElliottWave .NET library.

This module reproduces the behaviour of ``WnFElliottWave.dll`` (the
``ElliottWave`` / ``MacroWave`` classes) which performs Elliott Wave pattern
recognition and structure/probability notation on a price series.

The original library is a pure managed .NET assembly; all of the algorithmic
logic has been translated here.  The WinForms/COM/UI plumbing of the original
(``FlagsEditor``, localised attributes, events) is intentionally omitted or
reduced to plain Python callables because it has no effect on the computed
result.

Typical use::

    from wnf_elliottwave import MacroWave, EPAM

    mw = MacroWave()
    mw.plist = [100, 105, 101, 96, ...]
    mw.monoize()
    for wave in mw.analyze(EPAM.DI):
        print(wave.liS, wave.liE, wave.RC, wave.SP(), wave.mSP, wave.get_mDS())
"""

from __future__ import annotations

import math
from enum import IntEnum, IntFlag

__all__ = [
    "EW_RC",
    "EW_SP",
    "EW_NP",
    "EW_Structure",
    "EPAM",
    "L5Conditions",
    "ElliottWave",
    "MacroWave",
]


# ---------------------------------------------------------------------------
# C#-compatible helpers
# ---------------------------------------------------------------------------

INT_MAX = 0x7FFFFFFF
INT_MIN = -0x80000000


def _div(a: float, b: float) -> float:
    """Division with C# ``double`` semantics (no ZeroDivisionError)."""
    if b == 0.0:
        if a == 0.0:
            return float("nan")
        return math.inf if a > 0 else -math.inf
    return a / b


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EW_RC(IntEnum):
    R10 = 10
    R1A = 11
    R1B = 12
    R1C = 13
    R1D = 14
    R20 = 20
    R2A = 21
    R2B = 22
    R2C = 23
    R2D = 24
    R2E = 25
    R30 = 30
    R3A = 31
    R3B = 32
    R3C = 33
    R3D = 34
    R3E = 35
    R3F = 36
    R40 = 40
    R4A = 41
    R4B = 42
    R4C = 43
    R4D = 44
    R4E = 45
    R50 = 50
    R5A = 51
    R5B = 52
    R5C = 53
    R5D = 54
    R60 = 60
    R6A = 61
    R6B = 62
    R6C = 63
    R6D = 64
    R70 = 70
    R7A = 71
    R7B = 72
    R7C = 73
    R7D = 74


class EW_SP(IntFlag):
    NONE = 0
    Five = 1
    SpecialFive = 2
    LastFive = 4
    FirstThree = 8
    CenterThree = 0x10
    SecondToLastThree = 0x20
    LastThree = 0x40
    LegBThree = 0x80
    XWaveThree = 0x100
    HiddenS5x = 0x200
    Undefined = 0x400


class EW_NP(IntEnum):
    High = 0
    Low = 0x800
    Rare = 0x1000


class EW_Structure(IntEnum):
    Unknown = 0
    Five = 1
    Three = 8
    Flat = 0x200
    ZigZag = 0x400
    Triangle = 0x800
    Complex = 0x1000


class EPAM(IntEnum):
    DE = 0
    SI = 1
    DI = 2


class L5Conditions(IntFlag):
    NONE = 0
    LastReversal = 1
    FirstFive = 2
    ThirdFive = 4
    LengthConstraints = 8
    Tow2FourLineBreak = 0x10
    Failure = 0x20


# ---------------------------------------------------------------------------
# Notation string helpers
# ---------------------------------------------------------------------------

_SP_STRING = {
    EW_SP.Five: ":5",
    EW_SP.SpecialFive: ":s5",
    EW_SP.LastFive: ":L5",
    EW_SP.FirstThree: ":F3",
    EW_SP.CenterThree: ":c3",
    EW_SP.SecondToLastThree: ":sL3",
    EW_SP.LastThree: ":L3",
    EW_SP.LegBThree: "b:c3",
    EW_SP.XWaveThree: "x:c3",
    EW_SP.HiddenS5x: ":?○:?",
}


def sp_notation_string(n: int) -> str:
    return _SP_STRING.get(n, "")


def sp_to_string(n: int, p: EW_NP = EW_NP.High) -> str:
    if p == EW_NP.High:
        text = "{0}"
    elif p == EW_NP.Rare:
        text = "[{0}]"
    else:
        text = "({0})"
    s = _SP_STRING.get(n)
    if s is None:
        return ""
    return text.format(s)


def sp_notation(s: str) -> EW_SP:
    return {
        ":5": EW_SP.Five,
        ":s5": EW_SP.SpecialFive,
        ":L5": EW_SP.LastFive,
        ":F3": EW_SP.FirstThree,
        ":c3": EW_SP.CenterThree,
        ":sL3": EW_SP.SecondToLastThree,
        ":L3": EW_SP.LastThree,
        "b:c3": EW_SP.LegBThree,
        "x:c3": EW_SP.XWaveThree,
    }.get(s, EW_SP.NONE)


# ---------------------------------------------------------------------------
# ElliottWave
# ---------------------------------------------------------------------------

class ElliottWave:
    SP_NO_MATCH = "No match. Default labels used."

    def __init__(self, ew: "MacroWave | None" = None, e: int = INT_MAX, s: int = 0):
        self._sp: dict[int, list[str]] = {}
        self._np: dict[EW_SP, int] = {}
        self._liS = INT_MAX
        self._liE = INT_MIN
        self._liP = INT_MAX
        self._li0 = INT_MAX
        self._li2 = INT_MIN
        self._pws = 0
        self._pwe = 0
        self._mRC: str | None = None
        self._mSP: str = ""
        self._mHA: str = ""
        self._mDS: str | None = None
        self._ew: "MacroWave | None" = None
        self._rc: int = 0
        self._st: EW_NP = EW_NP.Rare
        self._nn: EW_SP = EW_SP.NONE
        self._le = True
        self._dn = True
        self._default_sp = False
        self._nfixed = False
        self._notified = False

        self._sp_fixed_cbs: list = []
        self._notify_cbs: list = []

        if ew is not None:
            if e < 1:
                raise ValueError("Wave End less than 1")
            if s < 0:
                raise ValueError("Wave Start less than 0")
            self._ew = ew
            self._liE = e
            self._liS = s

    # -- plain fields exposed with C# names ------------------------------
    liS = property(lambda self: self._liS, lambda self, v: setattr(self, "_liS", v))
    liE = property(lambda self: self._liE, lambda self, v: setattr(self, "_liE", v))
    liP = property(lambda self: self._liP, lambda self, v: setattr(self, "_liP", v))
    li0 = property(lambda self: self._li0, lambda self, v: setattr(self, "_li0", v))
    li2 = property(lambda self: self._li2, lambda self, v: setattr(self, "_li2", v))
    mRC = property(lambda self: self._mRC, lambda self, v: setattr(self, "_mRC", v))
    mSP = property(lambda self: self._mSP, lambda self, v: setattr(self, "_mSP", v))
    mHA = property(lambda self: self._mHA, lambda self, v: setattr(self, "_mHA", v))
    Linked = property(lambda self: self._le, lambda self, v: setattr(self, "_le", v))
    SPN_Bound = property(lambda self: getattr(self, "_spn_bound", EW_SP.NONE),
                         lambda self, v: setattr(self, "_spn_bound", v))
    PolywaveStart = property(lambda self: self._pws, lambda self, v: setattr(self, "_pws", v))
    PolywaveEnd = property(lambda self: self._pwe, lambda self, v: setattr(self, "_pwe", v))

    @property
    def RC(self):
        return self._rc

    @property
    def mDS(self):
        return self._mDS

    @mDS.setter
    def mDS(self, v):
        self._mDS = v

    @property
    def NotationFixed(self):
        return self._nfixed

    @NotationFixed.setter
    def NotationFixed(self, value):
        self._nfixed = value
        if (not self._nfixed) and self._sp_fixed_cbs:
            for cb in list(self._sp_fixed_cbs):
                cb(self, self._nfixed)

    def set_SPN(self, value: EW_SP):
        self._nn = value

    def add_SPNotationFixed(self, cb):
        self._sp_fixed_cbs.append(cb)

    def add_NotifyingNotation(self, cb):
        self._notify_cbs.append(cb)

    # -- core -----------------------------------------------------------
    def get_mDS(self, r: EW_SP = EW_SP.NONE) -> str:
        result = ""
        if r == EW_SP.NONE:
            if not self._mDS:
                self._mDS = self.SPDescriptions()
            result = self._mDS
        elif r in self._sp:
            result = self.SPDescriptions(r)
        return result

    def SP(self, r: EW_SP = EW_SP.NONE, strict: bool = True, prob: EW_NP = EW_NP.Rare) -> EW_SP:
        result = EW_SP.NONE
        if r == EW_SP.NONE:
            result = self._nn
        elif strict:
            if r == self._nn:
                result = r
        else:
            if prob == EW_NP.Rare:
                if r in self._sp or (int(r) + 2048) in self._sp or (int(r) + 4096) in self._sp:
                    result = r
            elif prob == EW_NP.Low:
                if r in self._sp or (int(r) + 2048) in self._sp:
                    result = r
            else:
                if r in self._sp:
                    result = r
        return result

    @staticmethod
    def GetRC(lb: float, lc: float, la: float) -> int:
        num = _div(la, lc)
        if num < -2.618:
            rc = EW_RC.R70
        elif num <= -1.618:
            rc = EW_RC.R60
        elif num <= -1.0:
            rc = EW_RC.R50
        elif num < -0.618:
            rc = EW_RC.R40
        elif num == -0.618:
            rc = EW_RC.R30
        elif not (num <= -0.382):
            rc = EW_RC.R10
        else:
            rc = EW_RC.R20

        if lb != 0.0:
            num = _div(lb, lc)
            if rc == EW_RC.R10:
                if num < -1.618:
                    rc = rc + 4
                elif num <= -1.0:
                    rc = rc + 3
                elif num <= -0.618:
                    rc = rc + 2
                else:
                    rc = rc + 1
            elif rc == EW_RC.R20:
                if num < -1.618:
                    rc = rc + 5
                elif num <= -1.0:
                    rc = rc + 4
                elif num <= -0.618:
                    rc = rc + 3
                elif num <= -0.382:
                    rc = rc + 2
                else:
                    rc = rc + 1
            elif rc == EW_RC.R30:
                if num < -2.618:
                    rc = rc + 6
                elif num <= -1.618:
                    rc = rc + 5
                elif num <= -1.0:
                    rc = rc + 4
                elif num <= -0.618:
                    rc = rc + 3
                elif num <= -0.382:
                    rc = rc + 2
                else:
                    rc = rc + 1
            elif rc == EW_RC.R40:
                if num < -2.618:
                    rc = rc + 5
                elif num <= -1.618:
                    rc = rc + 4
                elif num <= -1.0:
                    rc = rc + 3
                elif num <= -0.382:
                    rc = rc + 2
                else:
                    rc = rc + 1
            else:
                if num < -2.618:
                    rc = rc + 4
                elif num <= -1.618:
                    rc = rc + 3
                elif num <= -1.0:
                    rc = rc + 2
                else:
                    rc = rc + 1
        return int(rc)

    def Info(self) -> list[str]:
        return [
            self._rc_string(self._rc),
            self._mSP,
            self.get_mDS(),
            str(self._pws),
            str(self._pwe),
        ]

    @staticmethod
    def _rc_string(r: int) -> str:
        try:
            text = EW_RC(r).name
        except ValueError:
            text = str(int(r))
        if len(text) >= 3 and text[2].isdigit():
            return text[:2].replace("R", "R-")
        text = text[:2] + text[2:3].lower()
        return text.replace("R", "R-")

    def Count(self, strict: EW_NP = EW_NP.High, default_to_none: bool = True) -> None:
        mwl = self._ew.mw_list
        if self._liE in mwl:
            ew = mwl[self._liE]
            if self._liS == INT_MAX:
                self._liS = ew.liP
            if self._liS == INT_MAX or self._liS < 0:
                raise ValueError("Wave Start cannot be determined.")
            self._sp.clear()
            self._np.clear()
            self._nn = EW_SP.NONE
            self._mSP = ""
            self._get_rc()
            self._get_sp(strict, default_to_none)
            array = self.Info()
            ew.mRC = array[0]
            ew.mSP = array[1]
            ew.set_SPN(self._nn)
            if array[2] != "":
                ew.mDS = array[2]

    def UpdateSP(self, dt, cn: str) -> None:
        """``dt`` is any sequence supporting ``dt[liE][cn] = value``."""
        self.Count(EW_NP.Low)
        dt[self._liE][cn] = self._mSP
        if self._nfixed and self._sp_fixed_cbs:
            for cb in list(self._sp_fixed_cbs):
                cb(self, self._nfixed)
        if self.SPN_Bound == EW_SP.NONE:
            return
        if self.SP(self.SPN_Bound, strict=False, prob=EW_NP.Low) & self.SPN_Bound:
            if self._notify_cbs and not self._notified:
                for cb in list(self._notify_cbs):
                    cb(self, self.SPN_Bound)
                self._notified = True
        elif self._notified:
            for cb in list(self._notify_cbs):
                cb(self, -1 * int(self.SPN_Bound))
            self._notified = False

    def _sp_to_high(self, n: int):
        num = n
        if (num & 0x800) == 2048:
            num -= 2048
        elif (num & 0x1000) == 4096:
            num -= 4096
        p = n - num
        return EW_SP(num), EW_NP(p)

    def AddSP(self, s: str, *var_args: int) -> None:
        for num in var_args:
            eW_SP, p = self._sp_to_high(num)
            if num not in self._sp:
                if eW_SP in self._np:
                    self._np[eW_SP] |= int(p)
                else:
                    self._np[eW_SP] = int(p)
                self._nn |= eW_SP
                self._mSP = self._mSP + ("" if self._mSP == "" else "/") + sp_to_string(eW_SP, p)
            if num in self._sp:
                if s not in self._sp[num] and s != "":
                    self._sp[num].append(s)
                continue
            self._sp[num] = []
            if s is not None:
                self._sp[num].append(s)

    def RemoveSP(self, n: int) -> None:
        eW_SP, p = self._sp_to_high(n)
        if n in self._sp:
            del self._sp[n]
            self._mSP = self._mSP.replace(sp_to_string(eW_SP, p), "").replace("//", "/")
        if eW_SP in self._np:
            if (self._np[eW_SP] & int(p)) == int(p):
                self._np[eW_SP] -= int(p)
            if self._np[eW_SP] == 0 and eW_SP not in self._sp:
                del self._np[eW_SP]

    def RemoveSPLast(self, n: int) -> None:
        if n not in self._sp:
            return
        key, p = self._sp_to_high(n)
        if key in self._np and (self._np[key] & int(p)) == int(p):
            if len(self._sp[n]) <= 1:
                self.RemoveSP(n)
            else:
                self._sp[n].pop()

    # -- rule engine -----------------------------------------------------
    def _get_rc(self) -> None:
        candidate = False
        self._li0 = self._ew.w_m0s(self._liS, self._liE)
        self._li2, candidate = self._ew.w_m2e(self._liS, self._liE, candidate)
        if self._li2 > 0:
            lc = self._ew.plist[self._liE] - self._ew.plist[self._liS]
            la = self._ew.plist[self._li2] - self._ew.plist[self._liE]
            if self._li0 > 0 and self._li0 != INT_MAX:
                lb = self._ew.plist[self._liS] - self._ew.plist[self._li0]
            else:
                lb = 0.0
            self._rc = ElliottWave.GetRC(lb, lc, la)

    def _spr40(self, r: int) -> None:
        if r == EW_RC.R4A:
            self.mSP = ":F3/:c3/:s5/[:sL3]"
            self._nn = EW_SP.SpecialFive | EW_SP.FirstThree | EW_SP.CenterThree
            if self._st == EW_NP.Rare:
                self._nn |= EW_SP.SecondToLastThree
        elif r == EW_RC.R4B:
            self.mSP = ":F3/:c3/:s5/(:sL3)/(x:c3)/[:L5]"
            self._nn = EW_SP.SpecialFive | EW_SP.FirstThree | EW_SP.CenterThree
            if self._st == EW_NP.Rare:
                self._nn = self._nn | EW_SP.SecondToLastThree | EW_SP.XWaveThree | EW_SP.LastFive
            elif self._st == EW_NP.Low:
                self._nn = self._nn | EW_SP.SecondToLastThree | EW_SP.XWaveThree
        elif r == EW_RC.R4C:
            self.mSP = ":c3/(:F3)/(x:c3)"
            self._nn = EW_SP.CenterThree
            if self._st > EW_NP.High:
                self._nn = self._nn | EW_SP.FirstThree | EW_SP.XWaveThree
        elif r == EW_RC.R4D:
            self.mSP = ":F3/(:c3)/(x:c3)"
            self._nn = EW_SP.FirstThree
            if self._st > EW_NP.High:
                self._nn = self._nn | EW_SP.CenterThree | EW_SP.XWaveThree
        elif r == EW_RC.R4E:
            self.mSP = ":F3/(x:c3)/[:c3]"
            self._nn = EW_SP.FirstThree
            if self._st == EW_NP.Rare:
                self._nn = self._nn | EW_SP.XWaveThree | EW_SP.CenterThree
            elif self._st == EW_NP.Low:
                self._nn |= EW_SP.XWaveThree

    def _sp_default(self, rc: int) -> None:
        if self._dn:
            return
        base = int(rc) // 10 * 10
        if base == EW_RC.R10:
            self.mSP = ":5/(:c3)/(x:c3)/[:sL3]/[:s5]"
            self._nn = EW_SP.Five
            if self._st == EW_NP.Rare:
                self._nn = self._nn | EW_SP.CenterThree | EW_SP.XWaveThree | EW_SP.SecondToLastThree | EW_SP.SpecialFive
            elif self._st == EW_NP.Low:
                self._nn = self._nn | EW_SP.CenterThree | EW_SP.XWaveThree
        elif base == EW_RC.R20:
            self.mSP = ":5/(:sL3)/[:c3]/[:s5]"
            self._nn = EW_SP.Five
            if self._st == EW_NP.Rare:
                self._nn = self._nn | EW_SP.SecondToLastThree | EW_SP.CenterThree | EW_SP.SpecialFive
            elif self._st == EW_NP.Low:
                self._nn |= EW_SP.SecondToLastThree
        elif base == EW_RC.R30:
            self.mSP = ":F3/:c3/:s5/:5/(:sL3)/[:L5]"
            self._nn = EW_SP.Five | EW_SP.SpecialFive | EW_SP.FirstThree | EW_SP.CenterThree
            if self._st == EW_NP.Rare:
                self._nn = self._nn | EW_SP.SecondToLastThree | EW_SP.LastFive
            elif self._st == EW_NP.Low:
                self._nn |= EW_SP.SecondToLastThree
        elif base == EW_RC.R40:
            self._spr40(rc)
        elif base == EW_RC.R50:
            self.mSP = ":F3/:c3/:5/:L5/(:L3)"
            self._nn = EW_SP.Five | EW_SP.LastFive | EW_SP.FirstThree | EW_SP.CenterThree
            if self._st > EW_NP.High:
                self._nn |= EW_SP.LastThree

    def SPDescriptions(self, n: EW_SP = EW_SP.NONE) -> str:
        text = ""
        if n == EW_SP.NONE:
            for key in list(self._sp.keys()):
                text = text + " " + self.SPDescriptions(key)
        elif n in self._sp:
            if n != EW_SP.Undefined:
                text2 = sp_notation_string(n)
                if text2.startswith(":"):
                    text2 = text2[1:]
                if len(self._sp[n]) == 0:
                    text = "※ " + text2
                else:
                    for item in self._sp[n]:
                        text = text + " " + item.replace("※", "※ " + text2 + ":")
            else:
                for item2 in self._sp[n]:
                    text = text + " " + item2
        return text.strip()

    def _get_sp(self, strict: EW_NP = EW_NP.High, default_to_none: bool = True) -> None:
        self._st = strict
        self._dn = default_to_none
        rc = self._rc
        dispatch = {
            EW_RC.R1A: "SPR1A", EW_RC.R1B: "SPR1B", EW_RC.R1C: "SPR1C", EW_RC.R1D: "SPR1D",
            EW_RC.R2A: "SPR2A", EW_RC.R2B: "SPR2B", EW_RC.R2C: "SPR2C", EW_RC.R2D: "SPR2D",
            EW_RC.R2E: "SPR2E",
            EW_RC.R3A: "SPR3A", EW_RC.R3B: "SPR3B", EW_RC.R3C: "SPR3C", EW_RC.R3D: "SPR3D",
            EW_RC.R3E: "SPR3E", EW_RC.R3F: "SPR3F",
            EW_RC.R4A: "SPR4A", EW_RC.R4B: "SPR4B", EW_RC.R4C: "SPR4C", EW_RC.R4D: "SPR4D",
            EW_RC.R4E: "SPR4E",
            EW_RC.R5A: "SPR5A", EW_RC.R5B: "SPR5B", EW_RC.R5C: "SPR5C", EW_RC.R5D: "SPR5D",
            EW_RC.R6A: "SPR6A", EW_RC.R6B: "SPR6B", EW_RC.R6C: "SPR6C", EW_RC.R6D: "SPR6D",
            EW_RC.R7A: "SPR7A", EW_RC.R7B: "SPR7B", EW_RC.R7C: "SPR7C", EW_RC.R7D: "SPR7D",
        }
        name = dispatch.get(rc)
        if name is not None:
            getattr(self._ew, name)(self)
        if self.SP() == EW_SP.NONE:
            self._default_sp = True
            if not default_to_none:
                self._sp_default(self._rc)
                self._mDS = ElliottWave.SP_NO_MATCH + " " + self.get_mDS()

# ===========================================================================
# MacroWave
# ===========================================================================

class MacroWave:
    """Port of the ``WnFElliottWave.MacroWave`` class.

    It recognises the Elliott Wave structure of a price series and produces
    the structure/probability notation for every wave it can identify.
    """

    def __init__(self):
        self._plist: list = []
        self._mw_list = None
        self._ew_list = None
        self._update_cbs: list = []

    # -- properties -----------------------------------------------------
    plist = property(lambda self: self._plist, lambda self, v: setattr(self, "_plist", v))
    mw_list = property(lambda self: self._mw_list, lambda self, v: setattr(self, "_mw_list", v))
    ew_list = property(lambda self: self._ew_list, lambda self, v: setattr(self, "_ew_list", v))

    # -- notation helpers ----------------------------------------------
    @staticmethod
    def SPNotationString(n):
        return sp_notation_string(n)

    @staticmethod
    def SPNotation(s):
        return sp_notation(s)

    @staticmethod
    def GetMonowaves(p, ew):
        num = len(p) - 1
        if num < 1:
            return None
        result = {}
        num2 = 0
        num3 = 1
        num4 = 2
        while num4 <= num:
            num5 = p[num3] - p[num2]
            if num5 == 0.0:
                num3 += 1
                num4 += 1
                continue
            hit_end = False
            while True:
                num6 = p[num4] - p[num3]
                if num6 != 0.0:
                    break
                num4 += 1
                if num4 <= num:
                    continue
                hit_end = True
                break
            if hit_end:
                num3 = num
                break
            if num5 * num6 > 0.0:
                num3 = num4
                num4 += 1
                continue
            elliottWave = ElliottWave(ew, num3, num2)
            if len(result) > 0:
                elliottWave.liP = list(result.values())[-1].liS
            result[num2] = elliottWave
            num2 = num3
            num3 = num4
            num4 += 1
            continue
        elliottWave = ElliottWave(ew, num3, num2)
        if len(result) > 0:
            elliottWave.liP = list(result.values())[-1].liS
        result[num2] = elliottWave
        return result

    # -- lifecycle ------------------------------------------------------
    def resize_plist(self):
        self._plist = self._plist + [0.0]

    def subscribe_new(self, e, s, arg=EW_SP.NONE, add_item=True):
        elliottWave = None
        self._ew_list.append(ElliottWave(self, e, s))
        self._update_cbs.append(self._ew_list[-1].UpdateSP)
        if len(self._ew_list) > 1:
            self._ew_list[-2].SPN_Bound = arg
        self._ew_list[-1].Linked = True
        self._ew_list[-1].add_SPNotationFixed(self.OnWaveNotationFixed)
        if add_item:
            elliottWave = ElliottWave()
            elliottWave.liS = e
            elliottWave.liP = s
            self._mw_list[elliottWave.liS] = elliottWave
        return elliottWave

    def unsubscribe_update(self, i):
        cb = self._ew_list[i].UpdateSP
        try:
            self._update_cbs.remove(cb)
        except ValueError:
            pass
        del self._ew_list[i]

    def monoize(self):
        array = None
        num = 0
        num2 = 0
        num3 = 0
        num4 = 0.0
        num5 = 0.0
        self._mw_list = MacroWave.GetMonowaves(self._plist, self)
        if self._mw_list is not None and len(self._mw_list) != 0:
            array = [[0.0, 0.0, 0.0] for _ in range(len(self._plist))]
            array[0][0] = self._plist[0]
            array[0][1] = self._plist[0]
            for i, value in enumerate(self._mw_list.values()):
                num = value.liS
                num2 = value.liE
                num3 = num2 - num
                num5 = self._plist[num2] - self._plist[num]
                for j in range(1, num3):
                    array[num + j][0] = self._plist[num + j]
                    array[num + j][1] = self._plist[num] + j * num5 / num3
                array[num2][0] = self._plist[num2]
                array[num2][1] = self._plist[num2]
                if i > 1:
                    array[num2][2] = num5 / num4
                num4 = num5
        return array

    def analyze(self, rmode):
        self._ew_list = []
        if self._mw_list is None:
            return self._ew_list
        values = list(self._mw_list.values())
        num = self._mw_list[0].liS
        liE = self._mw_list[0].liE
        num2 = 1
        if rmode == EPAM.DE:
            for value2 in values:
                ew = ElliottWave(self, value2.liE, value2.liS)
                ew.add_SPNotationFixed(self.OnWaveNotationFixed)
                ew.Linked = True
                ew.Count(EW_NP.Low, True)
        elif rmode == EPAM.SI:
            for j in range(1, len(values)):
                value = values[j]
                if num2 == -1:
                    num2 = 1
                    value.mSP = "RN"
                    continue
                liE = value.liS
                if self.w_sim_nw(num, liE):
                    ew = ElliottWave(self, liE, num)
                    ew.add_SPNotationFixed(self.OnWaveNotationFixed)
                    ew.Linked = True
                    ew.Count(EW_NP.Low, True)
                    num = liE
                else:
                    num2 *= -1
        else:
            s = num
            s2 = 0
            num3 = 0
            candidate = False
            for i in range(1, len(values)):
                value = values[i]
                liS = value.liS
                if num2 == -1:
                    if liS == num3:
                        num2 = 1
                        num3, candidate = self.w_m2e(s2, num3, candidate)
                elif liS >= num3:
                    if self.w_dir_move_end(num, liS):
                        ew = ElliottWave(self, liS, s)
                        ew.add_SPNotationFixed(self.OnWaveNotationFixed)
                        ew.Linked = True
                        ew.Count(EW_NP.Low, True)
                        s = liS
                        num2 = 1
                    else:
                        s2 = liS
                        num3, candidate = self.w_m2e(num, liS, candidate)
                        num2 *= -1
                num = liS
        self._ew_list = sorted(self._ew_list, key=lambda w: w.liS)
        return self._ew_list

    # -- wave location helpers -----------------------------------------
    def w_m0s(self, s, e):
        if s < 0 or e <= s:
            raise ValueError("Invalid argument(s) %d, %d" % (s, e))
        if s not in self._mw_list:
            raise ValueError("No monowave starting at %d" % s)
        elliottWave = self._mw_list[s]
        num = self._plist[s]
        num2 = elliottWave.liP
        key = s
        num3 = 0
        flag = False
        flag2 = self._plist[s] < self._plist[e]
        if flag2:
            num4 = self._plist[e]
            num5 = self._plist[s]
        else:
            num4 = self._plist[s]
            num5 = self._plist[e]
        if elliottWave.liE == e:
            flag = True
            if elliottWave.liP < elliottWave.liS and (self._plist[elliottWave.liP] > num4 or self._plist[elliottWave.liP] < num5):
                elliottWave.li0 = elliottWave.liP
                return elliottWave.liP
            if elliottWave.li0 < elliottWave.liS and (self._plist[elliottWave.li0] > num4 or self._plist[elliottWave.li0] < num5):
                return elliottWave.li0
        key = self._mw_list[key].liP
        while key < s:
            num3 += 1
            num = self._plist[key]
            if num > num4 or num < num5:
                break
            if flag2:
                if num >= self._plist[num2]:
                    num2 = key
            elif num <= self._plist[num2]:
                num2 = key
            key = self._mw_list[key].liP
        if (num > num4 or num < num5) and num3 % 2 == 1:
            num2 = key
        if flag:
            elliottWave.li0 = num2
        return num2

    def w_m2e(self, s, e, candidate):
        if s < 0 or e <= s:
            raise ValueError("Invalid argument(s) %d, %d" % (s, e))
        elliottWave = None
        num = self._plist[e]
        num2 = e
        num3 = e
        num4 = 0
        flag = False
        flag2 = self._plist[s] < self._plist[e]
        if flag2:
            num5 = self._plist[e]
            num6 = self._plist[s]
        else:
            num5 = self._plist[s]
            num6 = self._plist[e]
        if s in self._mw_list:
            elliottWave = self._mw_list[s]
            if elliottWave.liE == e:
                flag = True
                if elliottWave.li2 > elliottWave.liE and (self._plist[elliottWave.li2] > num5 or self._plist[elliottWave.li2] < num6):
                    candidate = False
                    return elliottWave.li2, candidate
        while num3 in self._mw_list:
            num4 += 1
            num2 = num3
            num3 = self._mw_list[num3].liE
            num = self._plist[num3]
            if num > num5 or num < num6:
                break
            if flag2:
                if num <= self._plist[num2]:
                    num2 = num3
            elif num >= self._plist[num2]:
                num2 = num3
        candidate = candidate and (not (num > num5)) and (not (num < num6))
        if (num > num5 or num < num6) and num4 % 2 == 1:
            num2 = num3
        if flag:
            elliottWave.li2 = num2
        return num2, candidate

    # -- notation parsing ----------------------------------------------
    @staticmethod
    def _get_sp_notations_str(s, strict):
        eW_SP = EW_SP.NONE
        if s == "":
            return eW_SP
        s = s.replace("{}", "")
        for text in s.split("}{"):
            text = text.replace("{", "").replace("}", "").strip()
            if text == "":
                continue
            for text2 in text.split("/"):
                if text2 != "" and (not strict or (not text2.startswith("(") and not text2.startswith("["))):
                    eW_SP |= sp_notation(text2)
        return eW_SP

    def _get_sp_notations_idx(self, e, s, strict, default_to_none=True):
        elliottWave = ElliottWave(self, e, s)
        elliottWave.Linked = True
        elliottWave.Count(EW_NP.Low if not strict else EW_NP.High, default_to_none)
        elliottWave.Info()
        return elliottWave.SP()

    # -- higher level search -------------------------------------------
    def get_h1n(self, s, e, h1s, degree, max_=5):
        result = EW_Structure.Unknown
        dictionary = {}
        array = None
        array2 = None
        flag = False
        candidate = False
        degree2 = 0
        num = e
        h1s2 = s
        num2 = e
        h1s3 = 0
        num3 = 0
        num4 = 0
        eW_SP = EW_SP.NONE
        while h1s2 < num and num3 < max_:
            num3 += 1
            eW_SP = self._get_sp_notations_idx(num, h1s2, True)
            goto_185 = False
            if num3 == 1 and degree >= 0:
                ds = self._mw_list[num].get_mDS()
                if ds and ds.find("Running Correction") > 0:
                    flag = True
            elif num3 == 2:
                num4 = self.w_mw_cnt(h1s2, num)
                if num4 == 1 and self._mw_list[num].mSP.find("RN") > 0:
                    if num in dictionary:
                        del dictionary[num]
                    h1s2 = self.w_m0s(h1s2, num)
                    num = num2
                    num3 -= 2
                    continue
                while eW_SP != 0:
                    if eW_SP & EW_SP.LastThree:
                        spn = self._get_sp_notations_idx(h1s2, self.w_m0s(h1s2, num), True)
                        if (spn & EW_SP.CenterThree) or (spn & EW_SP.SecondToLastThree):
                            break
                        eW_SP = EW_SP(int(eW_SP) - 64)
                        continue
                    if eW_SP & EW_SP.XWaveThree:
                        eW_SP = EW_SP(int(eW_SP) - 256)
                        continue
                    goto_185 = True
                    break
            elif num3 == 3 or num3 == 5:
                num4 = self.w_mw_cnt(h1s2, num)
                num2 = num
                if eW_SP & EW_SP.SpecialFive:
                    degree2 = 0
                    array2, h1s2, degree2 = self.get_h1n(h1s2, num, h1s2, degree2, 3)
                    eW_SP = self._get_sp_notations_idx(num, h1s2, True)
                elif num4 > 1:
                    elliottWave = self._mw_list[num]
                    elliottWave.Linked = True
                    elliottWave.Count()
                    degree2 = -1
                    if elliottWave.SP() == EW_SP.LastFive:
                        array2, h1s3, degree2 = self.get_h1n(elliottWave.liP, elliottWave.liS, h1s3, degree2)
                        if h1s3 < h1s2:
                            h1s2 = h1s3
                    eW_SP = self._get_sp_notations_idx(num, h1s2, True)
                h1s3 = self.w_m0s(h1s2, num)
                num4 = self.w_mw_cnt(h1s3, h1s2)
                if self._mw_list[h1s2].mSP is not None and num4 == 1 and self._mw_list[h1s2].mSP.find("RN") > 0:
                    h1s2 = self.w_m0s(h1s3, h1s2)
                    num3 -= 1
                    continue
                if num3 == 3:
                    if degree2 > 0:
                        try:
                            result = EW_Structure[array2[2]]
                        except KeyError:
                            result = EW_Structure.Unknown
                    elif eW_SP != 0:
                        if not (eW_SP & EW_SP.Five) and (eW_SP & EW_SP.FirstThree):
                            result = EW_Structure.Flat
                            if not flag:
                                max_ = 3
                        elif eW_SP & EW_SP.Five:
                            result = EW_Structure.ZigZag
                    else:
                        result = EW_Structure.Unknown
            elif num3 == 4:
                h1s3, candidate = self.w_m2e(h1s2, num, candidate)
                if h1s3 > num2:
                    if not flag:
                        num3 -= 1
                        break
                elif eW_SP == EW_SP.NONE:
                    if ((result != EW_Structure.ZigZag) | (ElliottWave.GetRC(0.0, self._plist[num2] - self._plist[num], self._plist[num] - self._plist[h1s2]) >= EW_RC.R40)) and not flag:
                        num3 -= 1
                        break
                elif eW_SP & EW_SP.Five:
                    if not flag:
                        num3 -= 1
                        break
                elif not (eW_SP & EW_SP.FirstThree) and not (eW_SP & EW_SP.CenterThree):
                    if not flag:
                        num3 -= 1
                        break
                elif not (eW_SP & EW_SP.FirstThree) and (eW_SP & EW_SP.XWaveThree) and not flag:
                    num3 -= 1
                    break
            if goto_185:
                if eW_SP & EW_SP.LastFive:
                    degree2 = -1
                    array2, h1s3, degree2 = self.get_h1n(h1s2, num, h1s3, degree2, 3)
                    if h1s3 < h1s2:
                        h1s2 = h1s3
                        num3 -= 1
                        continue
                elif eW_SP & EW_SP.Five:
                    if not flag:
                        num3 -= 1
                        break
                elif not (eW_SP & EW_SP.FirstThree) and not (eW_SP & EW_SP.CenterThree) and not (eW_SP & EW_SP.SecondToLastThree) and not (eW_SP & EW_SP.LegBThree) and not flag:
                    num3 -= 1
                    break
            h1s3 = h1s2
            if num3 < max_:
                h1s2 = self.w_m0s(h1s2, num)
            num = h1s3
            dictionary[num] = int(eW_SP)
            continue
        source = sorted(dictionary.keys())
        h1s = source[0]
        elliottWave2 = ElliottWave(self, e, h1s)
        elliottWave2.Linked = True
        elliottWave2.Count(EW_NP.Low)
        array = elliottWave2.Info()
        if len(dictionary) == 1:
            return array, h1s, degree
        degree += 1
        if len(dictionary) == 3:
            array[2] = result.name
            if result == EW_Structure.ZigZag:
                eW_SP = self._get_sp_notations_idx(e, h1s, True)
                if eW_SP & EW_SP.LastFive:
                    array, h1s, degree = self.get_h1n(h1s, e, h1s, degree)
            return array, h1s, degree
        if flag:
            array[2] = "Running Correction"
        eW_SP = self._get_sp_notations_str(array[1], True)
        if eW_SP & EW_SP.LastFive:
            array, h1s, degree = self.get_h1n(h1s, e, h1s, degree)
        return array, h1s, degree

    @staticmethod
    def CriticalFwWaveCount(rc):
        if rc in (EW_RC.R1B, EW_RC.R5C, EW_RC.R5D):
            return 6
        if rc in (EW_RC.R1A, EW_RC.R3A, EW_RC.R3B, EW_RC.R3C, EW_RC.R3D, EW_RC.R4B, EW_RC.R4D, EW_RC.R4E):
            return 5
        if rc in (EW_RC.R1C, EW_RC.R2A, EW_RC.R2B, EW_RC.R2C, EW_RC.R2D, EW_RC.R3E, EW_RC.R3F, EW_RC.R4A, EW_RC.R4C, EW_RC.R5A, EW_RC.R5B, EW_RC.R6D, EW_RC.R7B, EW_RC.R7D):
            return 4
        if rc in (EW_RC.R1D, EW_RC.R2E, EW_RC.R6A, EW_RC.R6B, EW_RC.R6C, EW_RC.R7A, EW_RC.R7C):
            return 3
        return INT_MAX

    @staticmethod
    def CriticalBwWaveCount(rc):
        if rc in (EW_RC.R2C, EW_RC.R5A, EW_RC.R7C):
            return -4
        if rc in (EW_RC.R1C, EW_RC.R2D, EW_RC.R3A, EW_RC.R3D, EW_RC.R3E, EW_RC.R3F, EW_RC.R4B, EW_RC.R5B, EW_RC.R5C, EW_RC.R6A):
            return -3
        if rc in (EW_RC.R1A, EW_RC.R1B, EW_RC.R1D, EW_RC.R2A, EW_RC.R3C, EW_RC.R4A, EW_RC.R4C, EW_RC.R5D, EW_RC.R6B, EW_RC.R6C, EW_RC.R6D, EW_RC.R7A, EW_RC.R7B, EW_RC.R7D):
            return -2
        if rc in (EW_RC.R2B, EW_RC.R2E, EW_RC.R3B, EW_RC.R4D, EW_RC.R4E):
            return -1
        return INT_MIN

    def w_l5_check(self, li5, li3, li1, li6=None):
        l5Conditions = L5Conditions.NONE
        if li1 is not None and (li1.SP() & EW_SP.Five):
            l5Conditions |= L5Conditions.FirstFive
        if li3 is not None and (li3.SP() & EW_SP.Five):
            l5Conditions |= L5Conditions.ThirdFive
        if li1 is not None and li3 is not None and self.w_impulse_rule_l(self.w_l_v(li1.liS, li1.liE), self.w_l_v(li3.liS, li3.liE), self.w_l_v(li5.liS, li5.liE)):
            l5Conditions |= L5Conditions.LengthConstraints
        if li3 is not None:
            if li6 is not None:
                if self.w_tr_end(li3.liS, li5.liS, li5.liE, li6.liE):
                    l5Conditions |= L5Conditions.Tow2FourLineBreak
            else:
                candidate = False
                m2e, candidate = self.w_m2e(li5.liS, li5.liE, candidate)
                if self.w_tr_end(li3.liS, li5.liS, li5.liE, m2e):
                    l5Conditions |= L5Conditions.Tow2FourLineBreak
                if self.w_steep_ret(li5.liS, li5.liE):
                    l5Conditions |= L5Conditions.LastReversal
        return l5Conditions

    def w_mn1_m0(self, s, e):
        elliottWave = None
        num = self.w_m0s(s, e)
        n_mom1 = EW_SP.NONE
        if num in self._mw_list:
            elliottWave2 = ElliottWave(self, s, num)
            elliottWave2.Count(EW_NP.Low)
            n_mom1 = elliottWave2.SP()
            num2 = self.w_m0s(num, s)
            if num2 in self._mw_list:
                elliottWave = ElliottWave(self, num, num2)
                elliottWave.Linked = False
                elliottWave.Count(EW_NP.Low)
        return elliottWave, n_mom1

    # -- events ---------------------------------------------------------
    def OnWaveNotationFixed(self, li, flag):
        if flag:
            cb = li.UpdateSP
            try:
                self._update_cbs.remove(cb)
            except ValueError:
                pass
            try:
                self._ew_list.remove(li)
            except ValueError:
                pass
        elif li not in self._ew_list:
            self._update_cbs.append(li.UpdateSP)
            self._ew_list.append(li)

    def RaiseUpdateSP(self, dt, cn):
        for cb in list(self._update_cbs):
            cb(dt, cn)

    # -- common rule tail ----------------------------------------------
    def _tail(self, elwave, array, array2, num, num2):
        elwave.PolywaveStart = array2[num]
        elwave.PolywaveEnd = array[num2]
        elwave.NotationFixed = (num2 >= MacroWave.CriticalFwWaveCount(elwave.RC)
                                and array[num2] < len(self._plist) - 1)
        return elwave.SP()

    # ==================================================================
    # Rule methods (SPR*) - direct translations of MacroWave.SPR* methods
    # ==================================================================

    def SPR1A(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(liS, liE)
        num4 = liE - liS
        num5 = 0

        def _il_03ad():
            if num2 >= 3:
                num11 = self.w_l_v(array[2], array[3])
                if num11 < 1.618 * num3 and self.w_steep_ret(array[2], array[3]):
                    elwave.AddSP("※ m1 may be part of a Complex Correction which will necessitate the use of an x-wave", 1)
                    if num >= 2 and self.w_l_v(array2[2], array2[1]) > self.w_l_v(array2[1], array2[0]):
                        elwave.AddSP("※ The x-wave is not at the end of m0", 1)
                    elif elwave.Linked and array[0] in self._mw_list:
                        self._mw_list[array[0]].AddSP("", 256)
                    elwave.AddSP("", 512)
                    if num11 < 0.618 * num3:
                        elwave.AddSP("※ The probabilities increase dramatically that the x-wave is hidden in the center of m1", 512)

        if num2 >= 2:
            num5 = array[2] - array[1]
            if num2 >= 3:
                if num5 >= num4 or num5 >= array[3] - array[2]:
                    elwave.AddSP("", 1)
                if num2 >= 4 and num >= 1:
                    num6 = self.w_rr(array2[1], array2[0], array[0])
                    d = self.w_rr(array2[0], array[0], array[1])
                    if 1.0 <= num6 and num6 <= 1.618 and self.w_almost_same(d, 0.618) and not self.w_byond(array2[0], array[0], array[4]):
                        elwave.AddSP("※ m1 may complete a Flat pattern whthin a Complex formation where m2 is an x-wave", 2)
                        if elwave.Linked and array[2] in self._mw_list:
                            self._mw_list[array[2]].AddSP("", 256)
            if self.w_mw_cnt(array2[0], array[0]) >= 3 and self.w_steep_ret(array2[0], array[0]):
                elwave.AddSP("※ m0 is probably the end of an important Elliott pattern", 1024)
            if self.w_p_similar(array2[0], array[0], array[1], array[2]) and self.w_t_similar(array2[0], array[0], array[1], array[2]):
                flag = False
                if num2 >= 3 and num >= 1:
                    num7 = self.w_l_v(array2[1], array2[0])
                    if num7 >= 1.618 * num3:
                        flag2 = self.w_steeper(array[2], array[3], array2[1], array2[0])
                        if flag2:
                            flag = True
                        if num2 >= 5 and not flag2 and self.w_steeper(array[2], array[5], array2[1], array2[0]):
                            flag = True
                        if flag:
                            elwave.AddSP("※ A Running Correction (any variation) is probably unfolding", 1, 4112)
                    else:
                        num8 = self.w_l_v(array2[0], array[0])
                        num9 = self.w_l_v(array[2], array[3])
                        if num7 > num8 and num9 >= 1.618 * num3:
                            flag = True
                        if num2 >= 5 and num7 > num8 and num9 < 1.618 * num3 and self.w_l_v(array[4], array[5]) >= 1.618 * num3:
                            flag = True
                        if flag:
                            if num >= 2:
                                num7 = self.w_l_v(array2[1], array2[0])
                                num10 = self.w_l_v(array2[2], array2[1])
                                if num10 >= num7 and elwave.Linked and array2[0] in self._mw_list:
                                    self._mw_list[array2[0]].AddSP("", 32)
                                if num2 >= 3 and num10 < num7 and self.w_l_v(array[2], array[3]) <= 1.618 * num3:
                                    elwave.AddSP("※ A Running Correction (complex Double Three variety) may be unfolding, which starts at mn2 and ends at m4", 1, 256)
                                    _il_03ad()
                                    return self._tail(elwave, array, array2, num, num2)
                            elwave.AddSP("※ A Running Correction (any variation), which concludes more than one pattern, might be under formation", 1, 16)
                _il_03ad()
                return self._tail(elwave, array, array2, num, num2)
            if not self.w_overlap(array2[0], array[0], array[1], array[2]) and num2 >= 3 and num >= 1 and self.w_impulse_rule_3(self.w_l_v(array2[1], array2[0]), num3, self.w_l_v(array[2], array[3])):
                elwave.AddSP("※ m1 may be part of a larger Trending Impulse pattern", 1)
        return self._tail(elwave, array, array2, num, num2)

    def SPR1B(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        elwave.AddSP("※ Structure and progress labels of m1 would be :5", 1)
        array = self.w_adj_ws(liS, liE, 6)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        if num2 >= 4 and num >= 1:
            num5 = self.w_rr(array2[1], array2[0], array[0])
            if 1.0 <= num5 and num5 <= 1.618 and not self.w_byond(array2[0], array[0], array[4]):
                elwave.AddSP("※ m1 may complete a Flat pattern within a Complex formation where m2 is an x-wave", 2)
                if elwave.Linked and array[2] in self._mw_list:
                    self._mw_list[array[2]].AddSP("", 256)
        num6 = self.w_rt_time(array2[0], array[0], array[1])
        if self.w_mw_cnt(array2[0], array[0]) >= 3 and 0 < num6 and num6 <= array[0] - array2[0]:
            elwave.AddSP("※ m0 might conclude an important Elliott pattern", 1024)
        if self.w_overlap(array2[0], array[0], array[1], array[2]) and num2 >= 3 and num >= 1:
            num7 = self.w_l_v(array2[1], array2[0])
            num8 = self.w_l_v(array[2], array[3])
            if self.w_steeper(array[2], array[3], array[0], array[1]):
                if num7 >= num4:
                    elwave.AddSP("", 32)
                elif num2 >= 4 and (not self.w_p_similar(array2[0], array[0], array[1], array[2]) or not self.w_t_similar(array2[0], array[0], array[1], array[2])):
                    num9 = array[3] - array[0]
                    num10 = self.w_rt_time(array[0], array[3], array[4])
                    if 0 < num10 and num10 <= 0.5 * num9:
                        elwave.AddSP("※ A 5th Extension Terminal pattern may have completed with m3", 16)
                    elif num2 >= 6:
                        num11 = self.w_rt_time(array[0], array[3], array[6])
                        if 0 < num11 and num11 <= 0.5 * num9:
                            elwave.AddSP("※ A 5th Extension Terminal pattern may have completed with m3", 16)
            if num8 < num4 and num7 >= num3 and self.w_impulse_rule_3(num7, num4, num8) and num2 >= 4:
                num12 = array[3] - array2[1]
                num13 = self.w_rt_time(array[0], array[3], array[4])
                if 0 < num13 and num13 <= 0.5 * num12:
                    elwave.AddSP("", 16)
                elif num2 >= 6:
                    num14 = self.w_rt_time(array[0], array[3], array[6])
                    if 0 < num14 and num14 <= 0.5 * num12:
                        elwave.AddSP("", 16)
        return self._tail(elwave, array, array2, num, num2)

    def SPR1C(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        elwave.AddSP("※ Structure and progress notation of m1 would be :5", 1)
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        d = self.w_l_v(array[1], array[2])
        num5 = array[0] - array2[0]
        num6 = liE - liS
        num7 = array[2] - array[1]
        if (num2 >= 3 and self.w_almost_same(num3, num4)
                and self.w_t_similar(array2[0], array[0], array[0], array[1])
                and self.w_steeper(array[2], array[3], array[0], array[1])
                and (num7 >= num5 or num7 >= num6)
                and self.w_almost_same(d, 0.382 * num4)
                and self._mw_list[array[0]].SP(EW_SP.FirstThree, strict=False) > EW_SP.NONE):
            elwave.AddSP("", 4112)
        if num2 >= 4 and num >= 3 and self.w_steeper(array[2], array[3], array[0], array[1]):
            num8 = self.w_rr(array[4], array[3], array[2])
            num9 = self.w_l_v(array2[2], array2[1])
            if ((1.0 < num8 or num8 <= 0.618)
                    and self.w_almost_same(d, 0.382 * num4)
                    and self._mw_list[array[0]].SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE
                    and self.w_l_v(array2[3], array2[2]) > num9
                    and (num9 > num3 or self.w_l_v(array2[1], array2[0]) > num3)):
                elwave.AddSP("", 2080)
        return self._tail(elwave, array, array2, num, num2)

    def SPR1D(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        elwave.AddSP("※ There is only one possibility", 1)
        array = self.w_adj_ws(liS, liE, 3)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        return self._tail(elwave, array, array2, num, num2)

    def SPR2A(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        elwave.AddSP("※ Structure and progress labels of m1 would be :5", 1)
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        d = self.w_l_v(array[1], array[2])
        num5 = array[0] - array2[0]
        if num2 >= 4:
            if not self.w_byond(array2[0], array[0], array[4]):
                elwave.AddSP("※ m1 may be completing a Correction within a Complex formation where m2 is an x-wave", 2)
                if elwave.Linked and array[2] in self._mw_list:
                    self._mw_list[array[2]].AddSP("", 256)
            if num >= 1 and self.w_impulse_rule(self.w_l_v(array2[1], array2[0]), num4, self.w_l_v(array[2], array[3]), self.w_l_v(array[3], array[4])):
                elwave.AddSP("※ The market may be forming an Impulse pattern with m1 the center phase", 1)
        num6 = self.w_rt_time(array2[0], array[0], array[1])
        if self.w_mw_cnt(array2[0], array[0]) >= 3 and 0 < num6 and num6 <= num5:
            elwave.AddSP("※ m0 is probably the end of an important Elliott pattern", 1024)
        if self.w_Fibonacci_61_8(num3, d) and self.w_t_similar(array2[0], array[0], array[1], array[2]):
            flag = False
            if num >= 1:
                num7 = self.w_l_v(array2[1], array2[0])
                if num2 >= 3 and num7 >= 1.618 * num4 and self.w_steeper(array[2], array[3], array2[1], array2[0]):
                    flag = True
                    elwave.AddSP("※ A Running Correction (any variation) is probably unfolding which most likely started with m0 and concluded with m2", 4112)
                if flag and num2 >= 3:
                    num8 = self.w_l_v(array[2], array[3])
                    if self.w_almost_same(num5, array[2] - array[1]) and num8 < 1.618 * num4 and num7 > num3:
                        elwave.AddSP("※ m1 might be part of a Complex Correction which will necessitate the use of an x-wave", 1024)
                        if num >= 2 and self.w_l_v(array2[2], array2[1]) < num7:
                            elwave.AddSP("※ The x-wave might be at the end of m0", 1024)
                            if elwave.Linked and array[0] in self._mw_list:
                                self._mw_list[array[0]].AddSP("", 256)
                        if num2 >= 4 and self.w_l_v(array[3], array[4]) <= 1.618 * num8:
                            elwave.AddSP("※ The x-wave may be at the end of m2", 1024)
                            if elwave.Linked and array[2] in self._mw_list:
                                self._mw_list[array[2]].AddSP("", 256)
                        if num3 <= 0.5 * num4 and num4 > max(num7, num8):
                            elwave.AddSP("※ The x-wave may be hidden in the center of m1, which is most rare case", 512)
        if num2 >= 3 and num >= 1:
            num9 = self.w_l_v(array2[1], array2[0])
            if self.w_impulse_rule_3(num9, num4, self.w_l_v(array[2], array[3])) and num9 > num3 and num3 < num4 and self.w_steep_ret(array[2], array[3]):
                elwave.AddSP("※ m1 may be 3 of a Terminal Impulse pattern", 16)
        return self._tail(elwave, array, array2, num, num2)

    def SPR2B(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        elwave.AddSP("※ Structure and progress labels of m1 would be :5", 1)
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -1)
        num = len(array2) - 1
        num2 = len(array) - 1
        d = self.w_l_v(array2[0], array[0])
        num3 = self.w_l_v(liS, liE)
        d2 = self.w_l_v(array[1], array[2])
        num4 = array[0] - array2[0]
        num5 = array[2] - array[1]
        if num2 >= 4 and not self.w_byond(array2[0], array[0], array[4]):
            elwave.AddSP("※ m1 may complete a Correction within a Complex formation where m2 is an x-wave", 2)
            if elwave.Linked and array[2] in self._mw_list:
                self._mw_list[array[2]].AddSP("", 256)
        num6 = self.w_rt_time(array2[0], array[0], array[1])
        if self.w_mw_cnt(array2[0], array[0]) >= 3 and 0 < num6 and num6 <= num4:
            elwave.AddSP("※ m0 is probably the end of an important Elliott pattern", 1024)
        if self.w_p_similar(array2[0], array[0], array[1], array[2]) and self.w_t_similar(array2[0], array[0], array[1], array[2]):
            flag = False
            if num2 >= 3 and num >= 1 and self.w_l_v(array2[1], array2[0]) >= 1.618 * num3 and self.w_steeper(array[2], array[3], array2[1], array2[0]):
                flag = True
                elwave.AddSP("※ A Running Correction (any variation) is probably unfolding which most likely started with m0 and concluded with m2", 4112)
            if num2 >= 4 and flag:
                num7 = self.w_rt_time(array[2], array[3], array[4])
                num8 = array[3] - array[2]
                num9 = self.w_l_v(array[2], array[3])
                if self.w_almost_same(d, d2) and self.w_almost_same(num4, num5) and num9 < 1.618 * num3 and 0 < num7 and num7 <= num8:
                    elwave.AddSP("※ m1 may be part of a Complex Correction which will necessitate the use of an x-wave", 512)
                    if num9 <= 0.618 * num3:
                        elwave.AddSP("※ The probabilities increase dramatically that the x-wave, if any, is hidden in the center of m1", 512)
                    elif elwave.Linked:
                        if array[0] in self._mw_list:
                            self._mw_list[array[0]].AddSP("", 256)
                        if array[2] in self._mw_list:
                            self._mw_list[array[2]].AddSP("", 256)
        if (num2 >= 3 and num >= 1
                and (0.618 < _div(num5, num4) or _div(num4, num5) > 0.618)
                and self.w_overlap(array2[0], array[0], array[1], array[2])
                and self.w_impulse_rule_3(self.w_l_v(array2[1], array2[0]), num3, self.w_l_v(array[2], array[3]))
                and self.w_steep_ret(array[0], array[3])):
            elwave.AddSP("※ There is a chance m1 is an :sL3 which is part of a Terminal pattern", 32)
        return self._tail(elwave, array, array2, num, num2)

    def SPR2C(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        elwave.AddSP("", 1)
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -4)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        self.w_l_v(array[1], array[2])
        if num2 >= 4 and not self.w_byond(array2[0], array[0], array[4]):
            elwave.AddSP("※ m1 may complete a Flat within a Complex formation where m2 is an x-wave", 2)
            if elwave.Linked and array[2] in self._mw_list:
                self._mw_list[array[2]].AddSP("", 256)
            if num >= 2:
                num5 = self.w_l_v(array2[1], array2[0])
                if self.w_l_v(array2[2], array2[1]) < num5:
                    if num >= 4 and self.w_l_v(array2[4], array2[3]) > self.w_l_v(array2[3], array2[2]):
                        elwave.AddSP("※ There may be a Complex formation where m0 is x", 1)
                        if elwave.Linked and array[0] in self._mw_list:
                            self._mw_list[array[0]].AddSP("", 256)
                elif num4 >= 0.382 * num5:
                    elwave.AddSP("※ There may be a Complex formation where m2 is x", 2)
        num6 = self.w_rt_time(array2[0], array[0], array[1])
        if self.w_mw_cnt(array2[0], array[0]) >= 3 and 0 < num6 and num6 <= array[0] - array2[0]:
            elwave.AddSP("※ m0 probably ends an important Elliott pattern", 1024)
        if num2 >= 3 and num >= 1:
            num7 = self.w_l_v(array2[1], array2[0])
            num8 = self.w_l_v(array[2], array[3])
            if num7 > num3 and num7 < 2.618 * num4 and num4 > num8 and self.w_steep_ret(array[0], array[3]):
                elwave.AddSP("※ A Terminal pattern may have completed with m3", 16)
            if self.w_steep_ret(array[1], array[2]) and self.w_steeper(array[2], array[3], array[0], array[1]) and num7 <= 1.618 * num4:
                elwave.AddSP("※ A Running Triangle may have terminated with m2", 32)
                if num2 >= 4:
                    num9 = array[3] - array[2]
                    num10 = array[4] - array[3]
                    num11 = self.w_rt_time(array[2], array[3], array[4])
                    num12 = self.w_l_v(array[3], array[4])
                    if 0 < num11 and num11 <= num9:
                        elwave.AddSP("※ The Triangle is of the Limiting variety", 32)
                    if self.w_rr(array[4], array[3], array[2]) < 1.0 or (_div(num12, num10) < _div(num8, num9) and self.w_total_ret(array[3], array[4]) > 0):
                        elwave.AddSP("※ The Triangle is probably of the Non-Limiting variety or m3 will become part of a 5 segment Terminal formation", 32)
            if num7 > 1.618 * num4 and num8 > 1.618 * num4:
                elwave.AddSP("※ An Irregular Failure may have completed with m2", 16)
        return self._tail(elwave, array, array2, num, num2)

    def SPR2D(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        self.w_l_v(array[1], array[2])
        num5 = array[0] - array2[0]
        num6 = liE - liS
        num7 = array[2] - array[1]
        if num2 >= 3:
            num8 = self.w_l_v(array[2], array[3])
            num9 = array[3] - array[2]
            if num6 <= num7 or num7 >= num9:
                elwave.AddSP("", 1)
            if self.w_steeper(array[2], array[3], array[0], array[1]):
                if (self.w_steep_ret(array[1], array[2])
                        and self.w_t_similar(array2[0], array[0], array[0], array[1])
                        and (self.w_Fibonacci_61_8(num7, num5) or num7 > 0.618 * num5 or self.w_Fibonacci_61_8(num7, num6) or num7 > 0.618 * num6)
                        and num3 < 1.382 * num4):
                    elwave.AddSP("※ A severe c-Failure Flat may have concluded with m2", 16)
                if num2 >= 4 and num >= 3:
                    num10 = self.w_l_v(array2[2], array2[1])
                    num11 = self.w_rr(array[4], array[3], array[2])
                    if ((num11 > 1.0 or num11 <= 0.618)
                            and self.w_l_v(array2[3], array2[2]) > num10
                            and (num10 > num3 or self.w_l_v(array2[1], array2[0]) > num3)
                            and self._mw_list[array[0]].SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE):
                        elwave.AddSP("※ m1 may be the second-to-last leg of a Contracting Triangle", 2080)
            elif num2 >= 4 and num8 < num4 and self.w_rr(array[4], array[3], array[2]) >= 0.618 and num6 < num5 and num7 >= num6:
                elwave.AddSP("※ m1 is probably part of a Zigzag which concludes with m3", 1)
        return self._tail(elwave, array, array2, num, num2)

    def SPR2E(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        elwave.AddSP("※ No matter what the circumstances, :5 is a very likely structure label for m1", 1)
        array = self.w_adj_ws(liS, liE, 3)
        array2 = self.w_adj_ws(liS, liE, -1)
        num = len(array2) - 1
        num2 = len(array) - 1
        if num2 >= 3 and num >= 1 and self.w_steep_ret(array[1], array[2]) and self.w_steeper(array[2], array[3], array[0], array[1]) and not self.w_overlap(array2[1], array2[0], array[0], array[1]):
            elwave.AddSP("※ The market may have completed a Complex Correction at m2 with a missing x-wave in the middle of m0", 16)
            if elwave.Linked and array[0] in self._mw_list:
                self._mw_list[array[0]].AddSP("", 512)
        return self._tail(elwave, array, array2, num, num2)

    def SPR3A(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        candidate = False
        num3 = self.w_m0s(liS, liE)
        num4, candidate = self.w_m2e(liS, liE, candidate)
        num5, candidate = self.w_m2e(liE, num4, candidate)
        num6, candidate = self.w_m2e(num4, num5, candidate)
        b, candidate = self.w_m2e(num5, num6, candidate)
        num7 = self.w_m0s(num3, liS)
        num8 = self.w_m0s(num7, num3)
        a = self.w_m0s(num8, num7)
        num9 = self.w_l_v(num7, num3)
        num10 = self.w_l_v(a, num8)
        num11 = self.w_l_v(liS, liE)
        num12 = self.w_l_v(num4, num5)
        num13 = self.w_l_v(num5, num6)
        num14 = self.w_l_v(num6, b)
        num15 = num3 - num7
        num16 = liS - num3
        num17 = liE - liS
        if num12 > 2.618 * num11:
            if num11 > num9 and num11 > num10 and self.w_tr_end(num7, liS, liE, num4):
                elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
            if not (num9 > 1.618 * num11):
                elwave.AddSP("※ m1 could be the end of a zig-zag within a complex correction", 2050)
            elwave.AddSP("※ m1 most likely is the center portion of a running correction", 16)
            if num9 < 0.618 * num12:
                elwave.AddSP("※ More than one Elliott pattern may have completed at m2", 16)
        elif num12 >= 1.618 * num11:
            elwave.AddSP("※ m1 may be the center portion of a 5x impulse pattern", 2)
            elwave.AddSP("※ m1 may be the center portion of a running correction", 16)
            elwave.AddSP("※ m1 may be the first leg of a pattern within a complex correction", 8)
            if num11 > num9 and num11 > num10 and self.w_tr_end(num7, liS, liE, num4):
                elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
            if num9 > num12:
                elwave.RemoveSP(16)
            if num9 > num11:
                elwave.RemoveSP(2)
                elwave.AddSP("※ m1 could be the c-wave of a zig-zag within a complex correction; m2 would be an x-wave which is most likely followed by wave-a of a contracting triangle", 2)
        elif num12 >= num11:
            elwave.AddSP("※ m1 may be the first leg of a standard pattern within a complex correction", 8)
            elwave.AddSP("※ m1 may be the center portion of a 5x impulse pattern", 1)
            elwave.AddSP("※ m1 may be the c-wave of a zig-zag within an ongoing complex correction", 2)
            if num11 > num9 and num11 > num10 and self.w_tr_end(num7, liS, liE, num4):
                elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
            if num13 < num12:
                elwave.RemoveSP(8)
            if num16 < num15 and num16 < num17:
                elwave.RemoveSP(2)
            if num9 > num11 and elwave.SP(EW_SP.SpecialFive, strict=False) > EW_SP.NONE:
                elwave.RemoveSP(2)
                elwave.AddSP("※ m1 could only be the c-wave of a zig-zag within a complex correction; m2 would be an x-wave", 2)
        else:
            if self.w_steep_ret(num4, num5):
                elwave.AddSP("※ An impulse pattern may have concluded with m3", 1)
                elwave.AddSP("※ A complex corrective pattern may have concluded with m3", 8)
                if num11 > num9 and num11 > num10 and self.w_tr_end(num7, liS, liE, num4):
                    elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
            elif self.w_total_ret(num4, num5) > 0:
                elwave.AddSP("※ m1 concludes a zig-zag which is part of a complex corrective pattern", 2)
                if num11 > num9 and num11 > num10 and self.w_tr_end(num7, liS, liE, num4):
                    elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
            if num13 < num12:
                elwave.AddSP("※ m1 may have completed a zig-zag which is part of a complex correction", 2)
                if num11 > num9 and num11 > num10 and self.w_tr_end(num7, liS, liE, num4):
                    elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                if not (num14 > num12):
                    elwave.AddSP("※ m1 is part of a terminal impulsive pattern", 8)
        return self._tail(elwave, array, array2, num, num2)

    def SPR3B(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -1)
        num = len(array2) - 1
        num2 = len(array) - 1
        candidate = False
        num3 = self.w_m0s(liS, liE)
        num4, candidate = self.w_m2e(liS, liE, candidate)
        num5, candidate = self.w_m2e(liE, num4, candidate)
        num6, candidate = self.w_m2e(num4, num5, candidate)
        b, candidate = self.w_m2e(num5, num6, candidate)
        num7 = self.w_m0s(num3, liS)
        num8 = self.w_l_v(num7, num3)
        num9 = self.w_l_v(liS, liE)
        num10 = self.w_l_v(num4, num5)
        num11 = self.w_l_v(num5, num6)
        num12 = self.w_l_v(num6, b)
        if num10 > 2.618 * num9:
            elwave.AddSP("※ m1 most likely is the center portion of an irregular failure pattern", 16)
            elwave.AddSP("※ m1 could be wave-c of a zig-zag within a complex correction", 2050)
            if num8 > 1.618 * num9:
                elwave.RemoveSP(2050)
            if num8 < 0.618 * num10:
                elwave.AddSP("※ One or more Elliott patterns may end at m2", 16)
        elif num10 >= 1.618 * num9:
            elwave.AddSP("※ m1 may be the c-wave of a zig-zag within a complex correction", 2)
            if not (num8 > num9):
                elwave.AddSP("※ m1 may be the center portion of a 5x terminal impulse pattern", 16)
            elwave.AddSP("※ m1 may be the center portion of an irregular failure flat", 16)
        elif num10 >= num9:
            elwave.AddSP("※ m1 may be the first leg of a zig-zag within a complex correction", 1)
            elwave.AddSP("※ m1 may be the last leg of a zig-zag within a complex correction", 2)
            elwave.AddSP("※ m1 may be the center portion of a 5x terminal impulse pattern", 16)
            if num11 < num10:
                elwave.RemoveSP(1)
            if self.w_steep_ret(num4, num5):
                elwave.RemoveSP(2)
            if num8 > num9:
                elwave.RemoveSP(16)
                if elwave.SP(EW_SP.SpecialFive, strict=False) == EW_SP.SpecialFive and elwave.Linked and num4 in self._mw_list:
                    self._mw_list[num4].AddSP("", 256)
        else:
            if self.w_steep_ret(num4, num5):
                elwave.AddSP("※ m3 may complete a complex correction", 1)
                if num8 < 2.618 * num9 and self.w_rt_over(num7, num3, num6) < num5 - num7:
                    elwave.AddSP("※ m1 may be a part of a terminal impulse pattern", 16)
            elif self.w_total_ret(num4, num5) > 0:
                elwave.AddSP("※ m1 may conclude a zig-zag within a complex correction", 2)
            if num11 < num10:
                elwave.AddSP("※ m1 may be wave-c of a zig-zag within a complex correction", 2)
                if not (num12 > num10):
                    elwave.AddSP("※ m1 may be part of a terminal impulse pattern", 8)
        return self._tail(elwave, array, array2, num, num2)

    def SPR3C(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        candidate = False
        num3 = self.w_m0s(liS, liE)
        num4, candidate = self.w_m2e(liS, liE, candidate)
        num5, candidate = self.w_m2e(liE, num4, candidate)
        num6, candidate = self.w_m2e(num4, num5, candidate)
        num7, candidate = self.w_m2e(num5, num6, candidate)
        num8 = self.w_m0s(num3, liS)
        a = self.w_m0s(num8, num3)
        num9 = self.w_l_v(num8, num3)
        num10 = self.w_l_v(a, num8)
        num11 = self.w_l_v(liS, liE)
        num12 = self.w_l_v(num4, num5)
        num13 = self.w_l_v(num5, num6)
        num14 = self.w_l_v(num6, num7)
        if num12 > 2.618 * num11:
            if num9 <= 1.618 * num11:
                elwave.AddSP("※ m2 probably completed a non-limiting triangle", 32)
            if num9 > 1.618 * num11 or num10 > 0.618 * num9:
                elwave.AddSP("※ m2 probably completed an irregular failure flat", 16)
        elif num12 >= 1.618 * num11:
            elwave.AddSP("※ m1 may be the center portion of an irregular failure flat", 16)
            elwave.AddSP("※ m1 may be the second-to-last leg of a contracting triangle", 32)
            elwave.AddSP("※ m1 may be the part of a complex correction", 8, 2)
            if self.w_steep_ret(num4, num5):
                elwave.RemoveSP(2)
            if num9 > 1.618 * num11:
                elwave.RemoveSP(32)
            elif self.w_rr(liS, num3, num8) > 0.618:
                elwave.RemoveSP(16)
            if num13 < num12:
                elwave.RemoveSP(8)
        elif num12 >= num11:
            elwave.AddSP("※ m1 may be the center portion of an irregular failure flat", 16)
            elwave.AddSP("※ m1 may be the second-to-last leg of a contracting triangle", 32)
            elwave.AddSP("※ m1 may be the center leg of a 5x terminal pattern", 16)
            elwave.AddSP("※ m1 may be one of the legs of a complex correction", 8, 2)
            if num13 < num12:
                elwave.RemoveSP(8)
                elwave.RemoveSPLast(16)
            if self.w_steep_ret(num4, num5):
                elwave.RemoveSPLast(2)
            if num9 > 1.618 * num11:
                elwave.RemoveSP(32)
            elif self.w_rr(liS, num3, num8) > 0.618:
                elwave.RemoveSP(16)
        else:
            if self.w_steep_ret(num4, num5):
                elwave.AddSP("※ m3 may complete a complex correction", 8)
                if num9 < 1.382 * num11 or num9 > 2.618 * num11:
                    elwave.AddSP("※ m3 may complete a terminal impulse pattern", 4112)
                else:
                    elwave.AddSP("※ m3 may complete a terminal impulse pattern", 16)
            elif self.w_total_ret(num4, num5) > 0:
                elwave.AddSP("※ m1 may be wave-a of a zig-zag", 8)
                num15 = self.w_total_ret(num6, num7)
                if 0 >= num15 or num15 >= num6 - num5:
                    elwave.AddSP("※ m1 may be wave-c of a zig-zag within a complex correction", 2050)
            if num13 < num12:
                elwave.AddSP("※ m1 may be the last leg of a zig-zag or flat within a complex correction", 2)
                elwave.AddSP("※ m1 may be one of the center legs of a running contracting triangle", 16)
                elwave.AddSP("※ m1 may be the first leg of a terminal impulse pattern", 2056)
                if num14 > num12:
                    elwave.RemoveSPLast(8)
                if num9 > 2.618 * num11:
                    elwave.RemoveSPLast(2)
        return self._tail(elwave, array, array2, num, num2)

    def SPR3D(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        candidate = False
        num3 = self.w_m0s(liS, liE)
        num4, candidate = self.w_m2e(liS, liE, candidate)
        num5, candidate = self.w_m2e(liE, num4, candidate)
        num6, candidate = self.w_m2e(num4, num5, candidate)
        num7, candidate = self.w_m2e(num5, num6, candidate)
        num8 = self.w_m0s(num3, liS)
        s = self.w_m0s(num8, num3)
        a = self.w_m0s(s, num8)
        num9 = self.w_l_v(num8, num3)
        num10 = self.w_l_v(num3, liS)
        num11 = self.w_l_v(liS, liE)
        num12 = self.w_l_v(liE, num4)
        num13 = self.w_l_v(num4, num5)
        num14 = self.w_l_v(num5, num6)
        if num13 > 2.618 * num12:
            elwave.AddSP("※ m1 may be the first leg of a zig-zag", 1)
            elwave.AddSP("※ m1 may be the center section of a c-failure flat", 16)
            elwave.AddSP("※ m1 may be the second-to-last leg of a triangle", 2080)
            if num9 < 0.618 * num10 or num9 > 1.618 * num10:
                elwave.RemoveSP(2080)
            if self.w_total_ret(liE, num4) > num4 - liE:
                elwave.RemoveSP(2080)
                elwave.RemoveSP(16)
            if num13 > 1.618 * num11:
                elwave.RemoveSP(1)
        elif num13 >= 1.618 * num12:
            elwave.AddSP("※ m1 may be the center section of a c-failure flat", 16)
            elwave.AddSP("※ m1 may be the second-to-last leg of a contracting triangle", 32)
            elwave.AddSP("※ m1 may be the first leg of a zig-zag", 1)
            if num9 < 0.618 * num10 or num9 > 1.618 * num10:
                num15 = self.w_l_v(a, num3)
                if num11 < 0.382 * num15:
                    elwave.RemoveSP(32)
                elif num11 > 0.382 * num15 and num11 < 0.618 * num15:
                    elwave.RemoveSP(32)
                    elwave.AddSP("※ m1 may be the second-to-last leg of a contracting triangle", 2080)
            if num9 > 0.618 * num10 and num9 < 1.618 * num10:
                elwave.RemoveSP(16)
            if num14 < 0.618 * num10:
                elwave.RemoveSP(1)
                elwave.AddSP("※ m1 may be the first leg of a zig-zag", 2049)
        elif num13 >= num12:
            elwave.AddSP("※ m1 is probably the first leg a zig-zag", 1)
            elwave.AddSP("※ m1 may be a leg in a triangle", 2064, 4104)
            if num14 > num13:
                elwave.RemoveSP(2064)
                elwave.RemoveSP(4104)
            elif num14 < num13:
                num16 = self.w_rt_time(num5, num6, num7)
                if 0 < num16 and num16 <= num6 - num5 and self.w_steeper(num6, num7, liS, liE):
                    elwave.RemoveSP(1)
        return self._tail(elwave, array, array2, num, num2)

    def SPR3E(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_m0s(liS, liE)
        num4 = self.w_m0s(num3, liS)
        s = self.w_m0s(num4, num3)
        self.w_m0s(s, num4)
        candidate = False
        num5, candidate = self.w_m2e(liS, liE, candidate)
        num6, candidate = self.w_m2e(liE, num5, candidate)
        b, candidate = self.w_m2e(num5, num6, candidate)
        num7 = self.w_l_v(num4, num3)
        num8 = self.w_l_v(num3, liS)
        num9 = self.w_l_v(liS, liE)
        num10 = self.w_l_v(liE, num5)
        num11 = self.w_l_v(num5, num6)
        num12 = self.w_l_v(num6, b)
        if num11 > 2.618 * num10:
            elwave.AddSP("※ m1 may be the first leg of a zig-zag", 1)
            elwave.AddSP("※ m1 may be the center section of a c-failure flat which concludes a complex correction with a missing x-wave in the middle of m0", 16)
            elwave.AddSP("※ m1 may be the second-to-last leg of a triangle", 2080)
            if num7 < 0.618 * num8 or num7 > 1.618 * num8:
                elwave.RemoveSP(2080)
            if self.w_total_ret(liE, num5) > num5 - liE:
                elwave.RemoveSP(2080)
                elwave.RemoveSP(16)
            if num11 > 1.618 * num9:
                elwave.RemoveSP(1)
        elif num11 >= 1.618 * num10:
            elwave.AddSP("※ m1 may be the first leg of a zig-zag", 1)
            elwave.AddSP("※ m1 may be the center section of a c-failure flat which concludes a complex correction with a missing x-wave in the middle of m0", 16)
            if elwave.Linked and liS in self._mw_list:
                self._mw_list[liS].AddSP("", 512)
            if self.w_total_ret(liE, num5) > num5 - liE:
                elwave.RemoveSP(16)
            if num11 > 1.618 * num9:
                elwave.RemoveSP(1)
        elif num11 >= num10:
            elwave.AddSP("※ m1 may be the first leg of a zig-zag", 1)
            if self.w_mw_cnt(num6, b) != 1 or not (num12 > num11):
                elwave.AddSP("※ m1 may be the first leg of a triangle", 2056)
        return self._tail(elwave, array, array2, num, num2)

    def SPR3F(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_m0s(liS, liE)
        num4 = self.w_m0s(num3, liS)
        s = self.w_m0s(num4, num3)
        self.w_m0s(s, num4)
        candidate = False
        num5, candidate = self.w_m2e(liS, liE, candidate)
        num6, candidate = self.w_m2e(liE, num5, candidate)
        b, candidate = self.w_m2e(num5, num6, candidate)
        self.w_l_v(num4, num3)
        num7 = self.w_l_v(liS, liE)
        num8 = self.w_l_v(liE, num5)
        num9 = self.w_l_v(num5, num6)
        num10 = self.w_l_v(num6, b)
        if num9 > 2.618 * num8:
            elwave.AddSP("※ m1 may be the first leg of a zig-zag", 1)
            elwave.AddSP("※ m1 may be the center section of a c-failure flat which concludes a complex correction with a missing x-wave in the middle of m0", 2064)
            if self.w_total_ret(liE, num5) > num5 - liE:
                elwave.RemoveSP(2064)
            if elwave.SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE and not self.w_overlap(num4, num3, liS, liE) and elwave.Linked and liS in self._mw_list:
                self._mw_list[liS].AddSP("", 512)
            if num9 > 1.618 * num7:
                elwave.RemoveSP(1)
        elif num9 >= 1.618 * num8:
            elwave.AddSP("※ m1 may be the first leg of a zig-zag", 1)
            elwave.AddSP("※ m1 may be the center section of a c-failure flat which concludes a complex correction with a missing x-wave in the middle of m0", 2064)
            if num9 > num8:
                elwave.RemoveSP(2064)
            if self.w_total_ret(liE, num5) > num5 - liE:
                elwave.RemoveSP(2064)
            if elwave.SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE and not self.w_overlap(num4, num3, liS, liE) and elwave.Linked and liS in self._mw_list:
                self._mw_list[liS].AddSP("", 512)
            if num9 > 1.618 * num7:
                elwave.RemoveSP(1)
        elif num9 >= num8:
            elwave.AddSP("※ m1 may be the first leg of a zig-zag", 1)
            if self.w_mw_cnt(num6, b) != 1 or not (num10 > num9):
                elwave.AddSP("※ m1 may be the first leg of a triangle", 2056)
        return self._tail(elwave, array, array2, num, num2)

    def SPR4A(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(liS, liE)
        num4 = self.w_l_v(array[1], array[2])
        num5 = array[0] - array2[0]
        num6 = liE - liS
        num7 = array[2] - array[1]
        num8 = 0.0

        def _il_063e():
            return self._tail(elwave, array, array2, num, num2)

        def _il_0509():
            if self.w_steep_ret(array[2], array[3]):
                elwave.AddSP("※ There is virtually no chance m1 completed an Elliott pattern", 8, 16)
                if (num >= 1 and self.w_rr(array[2], array[1], array[0]) <= 0.7
                        and not self.w_overlap(array2[0], array[0], array[1], array[2])
                        and self.w_almost_same(num8, 1.618 * num3)
                        and (num5 > array2[0] - array2[1] or num5 > num6)):
                    elwave.AddSP("", 2)
                if not self.w_overlap(array2[0], array[0], array[1], array[2]):
                    elwave.RemoveSPLast(16)
                if elwave.SP(EW_SP.FirstThree, strict=False) > EW_SP.NONE:
                    elwave.AddSP("※ If :F3 is used, m1 would be wave-a of a correction within a larger complex formation, and m0 would be an x-wave", 8)
                if elwave.SP(EW_SP.SpecialFive, strict=False) > EW_SP.NONE:
                    elwave.AddSP("※ If :s5 is a possibility, m1 would be 3 of a 5th Extension Ipulse pattern", 2)
                if elwave.SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE:
                    elwave.AddSP("※ If :c3 is still a possibility, m1 may be part of a Expanding Triangle or a Terminal Impulse pattern", 16)
            return _il_063e()

        def _il_04bf():
            if elwave.SP(EW_SP.FirstThree, strict=False) > EW_SP.NONE:
                elwave.AddSP("※ If :F3 is chosen, m1 is wave-a of a Flat or Triangle", 8)
            if elwave.SP(EW_SP.SpecialFive, strict=False) > EW_SP.NONE:
                elwave.AddSP("※ If :s5 is appropriate, m1 is the end of a Zigzag", 2)
            return _il_0509()

        if num2 >= 3:
            num8 = self.w_l_v(array[2], array[3])
            if num8 > 2.618 * num4:
                if num >= 1 and self.w_l_v(array2[1], array2[0]) > 2.618 * num3:
                    elwave.AddSP("※ There is virtually no chance m1 is the end of any Elliott formation", 8)
                if num2 >= 4:
                    if self.w_rr(array[4], array[3], array[2]) > 1.0:
                        elwave.AddSP("※ It is extremely unlikely m1 is the end of any Elliott pattern", 8)
                    else:
                        elwave.AddSP("※ It is extremely unlikely m1 is the beginning of any Elliott pattern", 2)
            elif num8 >= 1.618 * num4:
                if num >= 1 and self.w_l_v(array2[1], array2[0]) > 2.618 * num3:
                    elwave.AddSP("※ There is virtually no chance m1 is the end of any Elliott formation", 8)
                if num2 >= 4:
                    if self.w_rr(array[4], array[3], array[2]) > 1.0:
                        elwave.AddSP("※ It is extremely unlikely m1 is the end of any Elliott pattern", 8)
                    else:
                        elwave.AddSP("", 2)
                        if self.w_rr(array[2], array[1], array[0]) > 0.7:
                            if self.w_overlap(array2[0], array[0], array[1], array[2]) and self.w_steep_ret(array[2], array[3]):
                                elwave.AddSP("※ m1 may be 3 of a 5x terminal Impulse pattern", 2)
                            else:
                                elwave.AddSP("※ It is most likely m1 terminated a Zigzag", 2)
                        elif num >= 1:
                            num9 = self.w_l_v(array2[1], array2[0])
                            if num3 > 2.618 * num9:
                                elwave.AddSP("※ The complex corrective scenario is the only proper choice", 2)
                            elif num3 >= 1.618 * num9:
                                elwave.AddSP("※ It becomes more likely that m1 is the end of a Zigzag inside of a complex correction with m2 the end of an x-wave; But the 3rd wave idea is still possible as long as you identify the 3rd wave as part of a Double Extension Impulse pattern with the 5th wave the longest", 2)
                            elif num3 >= num9:
                                if num >= 2 and self.w_l_v(array2[2], array2[1]) > num9 and self.w_overlap(array2[0], array[0], array[1], array[2]):
                                    elwave.AddSP("※ m1 may conclude the 3 of a terminal 5x impulse pattern", 2)
                            else:
                                elwave.AddSP("※ m1 can only be part of a Zigzag", 2)
            elif num8 >= num4:
                if num2 >= 4:
                    num10 = self.w_rt_time(array[2], array[3], array[4])
                    if 0 < num10 and num10 > array[3] - array[2]:
                        elwave.AddSP("※ m1 should be the first leg of a correction which follows an x-wave(m0)", 8)
                        elwave.AddSP("※ m1 should be end of a corrective phase which is part of a larger formation", 2)
                        if num >= 1:
                            if num3 < 0.618 * self.w_l_v(array2[1], array2[0]):
                                elwave.RemoveSP(2)
                            if num5 < array2[0] - array2[1] and num5 < num6:
                                elwave.RemoveSPLast(2)
                        if elwave.SP(EW_SP.FirstThree, strict=False) > EW_SP.NONE:
                            elwave.AddSP("※ If :F3 is chosen, m1 is wave-a of a Flat", 8)
                        if elwave.SP(EW_SP.SpecialFive, strict=False) > EW_SP.NONE:
                            elwave.AddSP("※ If :s5 is appropriate, m1 is the end of a Zigzag", 2)
                    if self.w_rr(array[4], array[3], array[2]) < 1.0:
                        elwave.AddSP("", 8)
                        if num >= 1:
                            if (self.w_mw_cnt(array[1], array[2]) >= 3 and self.w_steep_ret(array[1], array[2])
                                    and num7 > num6 and self.w_tr_end(array2[1], array[0], array[1], array[2])):
                                elwave.AddSP("※ m1 may be the end of a Zigzag within an Irregular or Running Correction", 4)
                            if num5 < array2[0] - array2[1] and num5 < num6:
                                return _il_04bf()
                        elwave.AddSP("", 2)
                        return _il_04bf()
                return _il_0509()
        return _il_063e()

    def SPR4B(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(liS, liE)
        if num2 >= 3:
            num4 = self.w_rr(array[3], array[2], array[1])
            num5 = self.w_rr(array[2], array[1], array[0])
            num6 = array[0] - array2[0]
            num7 = array[1] - array[0]
            num8 = array[2] - array[1]
            num9 = array[3] - array[2]
            if num4 > 2.618:
                if num >= 1:
                    num10 = self.w_l_v(array2[1], array2[0])
                    if num10 > 2.618 * num3:
                        elwave.AddSP("※ There is virtually no chance m1 is the end of any Elliott formation", 2056)
                        if num5 > 1.0:
                            elwave.AddSP("", 256)
                        else:
                            elwave.AddSP("※ There is virtually no chance m1 is the end of any Elliott formation", 16)
                    num11 = self.w_rt_time(array2[0], array[0], array[1])
                    if num10 >= 1.618 * num3 and 0 < num11 and num11 > array[0] - array2[0] and num7 >= 1.618 * num6:
                        elwave.AddSP("※ It is almost certain m0-m2 form an Irregular Failure Flat", 2056)
                        if num5 > 1.0:
                            elwave.AddSP("", 256)
                        else:
                            elwave.AddSP("※ It is almost certain m0-m2 form an Irregular Failure Flat", 16)
                    if num >= 3:
                        num12 = self.w_l_v(array2[3], array2[2])
                        if num3 >= num10 and num3 >= num12 and self.w_tr_end(array2[1], array[0], array[1], array[2]):
                            elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                if num2 >= 4 and self.w_rr(array[4], array[3], array[2]) < 0.618:
                    elwave.AddSP("※ An elongated flat begins with the start of m1", 8)
                    elwave.AddSP("※ It is extremely unlikely m1 is the beginning of any Elliott pattern", 2050)
                    if num >= 3:
                        num10 = self.w_l_v(array2[1], array2[0])
                        num12 = self.w_l_v(array2[3], array2[2])
                        if num3 >= num10 and num3 >= num12 and self.w_tr_end(array2[1], array[0], array[1], array[2]):
                            elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                    if num6 < array2[0] - array2[1] and num6 < num7:
                        elwave.RemoveSP(2050)
                    if num5 > 1.0:
                        elwave.AddSP("", 256)
                    else:
                        elwave.AddSP("※ It is extremely unlikely m1 is the beginning of any Elliott pattern", 16)
                if elwave.SP() == EW_SP.NONE:
                    elwave.AddSP("", 8, 16, 32, 2)
                    if num >= 3:
                        num10 = self.w_l_v(array2[1], array2[0])
                        num12 = self.w_l_v(array2[3], array2[2])
                        if num3 >= num10 and num3 >= num12 and self.w_tr_end(array2[1], array[0], array[1], array[2]):
                            elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                    if num >= 1:
                        num10 = self.w_l_v(array2[1], array2[0])
                        if num6 < array2[0] - array2[1] and num6 < num7:
                            elwave.RemoveSP(2)
                        if num3 < 0.618 * num10:
                            elwave.RemoveSP(2)
                    if num2 >= 4 and self.w_rr(array[4], array[3], array[2]) <= 0.618:
                        elwave.RemoveSP(8)
            elif num4 >= 1.618:
                if num >= 1:
                    num10 = self.w_l_v(array2[1], array2[0])
                    if num10 > 2.618 * num3:
                        elwave.AddSP("※ There is virtually no chance m1 is the end of any Elliott formation", 8, 16)
                    if num >= 3 and num3 >= num10 and num3 >= self.w_l_v(array2[3], array2[2]) and self.w_tr_end(array2[1], array[0], array[1], array[2]):
                        elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                if num2 >= 4 and self.w_rr(array[4], array[3], array[2]) <= 0.618:
                    elwave.AddSP("※ It is extremely unlikely m1 is the beginning of any Elliott pattern", 16)
                    elwave.AddSP("※ It is extremely unlikely m1 is the beginning of any Elliott pattern", 2080, 2050)
                    if num2 >= 5 and _div(self._plist[array[5]] - self._plist[array[2]], 1.618 * (self._plist[array[1]] - self._plist[array[0]])) < 1.0:
                        elwave.RemoveSP(32)
                    if num >= 1 and num6 < array2[0] - array2[1] and num6 < num7:
                        elwave.RemoveSP(2)
                    if not self.w_steep_ret(array[1], array[2]):
                        elwave.RemoveSP(32)
                    if elwave.SP(EW_SP.SecondToLastThree, strict=False) > EW_SP.NONE:
                        elwave.AddSP("※ If :sL3 is used, the Triangle (which concludes with m2) is Non-Limiting", 32)
                if elwave.SP() == EW_SP.NONE:
                    elwave.AddSP("", 8, 16, 32, 2)
                    if num5 > 1.0:
                        elwave.AddSP("", 256)
                    if num >= 3:
                        num10 = self.w_l_v(array2[1], array2[0])
                        if num3 >= num10 and num3 >= self.w_l_v(array2[3], array2[2]) and self.w_tr_end(array2[1], array[0], array[1], array[2]):
                            elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                    if not self.w_steep_ret(array[1], array[2]):
                        elwave.RemoveSP(32)
                    if num >= 1:
                        num10 = self.w_l_v(array2[1], array2[0])
                        if num6 < array2[0] - array2[1] and num6 < num7:
                            elwave.RemoveSP(2)
                        if num3 < 0.618 * num10:
                            elwave.RemoveSP(2)
                        if num3 <= num10 and num3 <= self.w_l_v(array[2], array[3]) and self.w_steep_ret(array[2], array[3]):
                            elwave.RemoveSP(16)
                    if num2 >= 4 and self.w_rr(array[4], array[3], array[2]) < 0.618:
                        elwave.RemoveSP(8)
            elif num4 >= 1.0:
                if self.w_steep_ret(array[2], array[3]):
                    elwave.AddSP("※ There is virtually no chance m1 is the end of any Elliott formation", 8, 16)
                    elwave.AddSP("※ If :c3 is the preferred choice, m1 may be part of a Terminal Impulse pattern", 16)
                    if num >= 3 and num3 >= self.w_l_v(array2[1], array2[0]) and num3 >= self.w_l_v(array2[3], array2[2]) and self.w_tr_end(array2[1], array[0], array[1], array[2]):
                        elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                if self.w_total_ret(array[2], array[3]) > num9:
                    elwave.AddSP("", 8, 16, 2)
                    if num >= 3 and num3 >= self.w_l_v(array2[1], array2[0]) and num3 >= self.w_l_v(array2[3], array2[2]) and self.w_tr_end(array2[1], array[0], array[1], array[2]):
                        elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                    if num >= 1:
                        num10 = self.w_l_v(array2[1], array2[0])
                        if num6 < array2[0] - array2[1] and num6 < num7:
                            elwave.RemoveSP(2)
                        if num3 < 0.618 * num10:
                            elwave.RemoveSP(2)
                        if num2 >= 4 and num10 >= 1.618 * num3 and self.w_rr(array[4], array[3], array[2]) < 0.618:
                            elwave.RemoveSPLast(8)
                if num2 >= 4:
                    if self.w_rr(array[4], array[3], array[2]) < 1.0:
                        elwave.AddSP("※ It is extremely unlikely m1 is the beginning of any Elliott pattern", 16, 2)
                        if (num >= 1 and self.w_mw_cnt(array[1], array[2]) >= 3 and self.w_steep_ret(array[1], array[2])
                                and num8 > num7 and self.w_l_v(array2[1], array2[0]) >= 1.618 * self.w_l_v(array2[0], array[0])
                                and self.w_tr_end(array2[1], array[0], array[1], array[2])):
                            elwave.AddSP("※ m1 may be the end of a Zigzag within an Irregular or Running Correction", 4)
                        if num >= 1 and num6 < array2[0] - array2[1] and num6 < num7:
                            elwave.RemoveSPLast(2)
                        if num >= 2:
                            num10 = self.w_l_v(array2[1], array2[0])
                            if self.w_l_v(array2[2], array2[1]) > num10:
                                elwave.RemoveSPLast(16)
                        if num2 >= 5 and not self.w_steep_ret(array[4], array[5]):
                            elwave.RemoveSPLast(16)
                        if elwave.SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE:
                            elwave.AddSP("※ If :c3 is still a possibility, m1 may be the x-wave of a Complex Correction", 256)
                    if self.w_rr(array[4], array[3], array[2]) < 0.618:
                        elwave.AddSP("※ It is extremely unlikely m1 is the beginning of any Elliott pattern", 16, 32, 2)
                        if num >= 3 and num3 >= self.w_l_v(array2[1], array2[0]) and num3 >= self.w_l_v(array2[3], array2[2]) and self.w_tr_end(array2[1], array[0], array[1], array[2]):
                            elwave.AddSP("※ m1 may be the 5 of a 5x impulse pattern", 4100)
                        if num2 >= 5 and _div(self._plist[array[5]] - self._plist[array[2]], 1.618 * (self._plist[array[1]] - self._plist[array[0]])) < 1.0:
                            elwave.RemoveSP(32)
                        if not self.w_steep_ret(array[1], array[2]):
                            elwave.RemoveSP(32)
                        if num >= 1 and num6 < array2[0] - array2[1] and num6 < num7:
                            elwave.RemoveSPLast(2)
                        if num >= 2:
                            num10 = self.w_l_v(array2[1], array2[0])
                            if self.w_l_v(array2[2], array2[1]) > num10:
                                elwave.RemoveSPLast(16)
                        if elwave.SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE:
                            elwave.AddSP("※ If :c3 is still a possibility, m1 may be the x-wave of a Complex Correction", 256)
        return self._tail(elwave, array, array2, num, num2)

    def SPR4C(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(liS, liE)
        num4 = self.w_l_v(array[1], array[2])
        if num2 >= 3:
            num5 = self.w_l_v(array[2], array[3])
            if num5 > 2.618 * num4:
                if self.w_steep_ret(array[1], array[2]):
                    elwave.AddSP("※ It is almost certain m1 is the center section of a c-Failure Flat or a Non-Limiting Contracting Triangle", 16)
                    elwave.AddSP("※ The only way [:F3] should be considered is it m3 is retraced more than 61.8% within a period of time equal to or less than m3", 4104)
                if num2 >= 4 and (not (self.w_rr(array[4], array[3], array[2]) >= 0.618) or array[4] - array[3] > array[3] - array[2]):
                    elwave.RemoveSP(4104)
            elif num5 >= 1.618 * num4:
                if self.w_steep_ret(array[1], array[2]) and num5 > 1.618 * num3:
                    elwave.AddSP("※ The chances are better that m1 was the center section of a c-Failure Flat or of a Contracting Triangle", 16)
                    elwave.AddSP("※ If the lower probability (:F3) is correct, m1 is part of an Elongated Flat", 2056)
                else:
                    elwave.AddSP("", 8, 16, 256)
            elif num5 >= num4:
                elwave.AddSP("", 8, 16)
        return self._tail(elwave, array, array2, num, num2)

    def SPR4D(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -1)
        num = len(array2) - 1
        num2 = len(array) - 1
        val = self.w_l_v(liS, liE)
        num3 = self.w_l_v(array[1], array[2])
        num4 = liE - liS
        if num2 >= 3:
            num5 = self.w_l_v(array[2], array[3])
            num6 = array[3] - array[2]
            if num5 > 2.618 * num3:
                if num6 <= num4 or self.w_steep_ret(array[1], array[2]):
                    elwave.AddSP("※ There is a strong possibility of a missing x-wave hidden toward the middle of m0", 16)
                    if elwave.Linked and liS in self._mw_list:
                        self._mw_list[liS].AddSP("", 512)
                if num2 >= 4 and self.w_rr(array[4], array[3], array[2]) >= 0.618:
                    elwave.AddSP("※ It is possible m1 is the first leg of a Flat", 8)
                if num6 > num4:
                    elwave.AddSP("", 8, 16)
            elif num5 >= num3:
                if self.w_steep_ret(array[1], array[2]):
                    if num2 >= 4 and self.w_rr(array[4], array[3], array[2]) < 0.618:
                        if _div(self._plist[array[3]] - self._plist[array[2]], 1.618 * (self._plist[array[1]] - self._plist[array[0]])) >= 1.0 and num6 <= num4:
                            elwave.AddSP("※ m1 may be part of a Complex Correction where the x-wave is hidden in m0", 8)
                            elwave.AddSP("※ m1 may be part of a Complex Correction where the x-wave is hidden in m0", 4112)
                            if elwave.Linked and liS in self._mw_list:
                                self._mw_list[liS].AddSP("", 512)
                                self._mw_list[liS].AddSP("", 8)
                        elif num2 >= 5:
                            num7 = array[5] - array[2]
                            if _div(self._plist[array[5]] - self._plist[array[2]], 1.618 * (self._plist[array[1]] - self._plist[array[0]])) >= 1.0 and num7 <= num4:
                                elwave.AddSP("※ m1 may be part of a Complex Correction where the x-wave is hidden in m0", 8)
                                elwave.AddSP("※ m1 may be part of a Complex Correction where the x-wave is hidden in m0", 4112)
                                if elwave.Linked and liS in self._mw_list:
                                    self._mw_list[liS].AddSP("", 512)
                                    self._mw_list[liS].AddSP("", 8)
                elif self.w_total_ret(array[1], array[2]) > array[2] - array[1]:
                    elwave.AddSP("※ A Flat or Triangle is probably involved", 8, 16)
                if self.w_steep_ret(array[2], array[3]):
                    elwave.AddSP("※ :F3 is the only reasonable choice for m1", 8)
                if num2 >= 4:
                    num8 = self.w_rr(array[4], array[3], array[2])
                    if num8 >= 0.618 and num8 < 1.0:
                        elwave.AddSP("※ :F3 is the only reasonable choice for m1", 8)
                    if num8 < 0.618:
                        elwave.AddSP("", 8)
                        if num2 >= 5:
                            if self.w_l_v(array[4], array[5]) < max(val, num5):
                                if self.w_steep_ret(array[4], array[5]):
                                    elwave.AddSP("※ m1 may be part of a Terminal pattern", 8)
                            else:
                                elwave.AddSP("※ m1 may be part of a Complex Double Flat where m4 is an x-wave", 8)
        return self._tail(elwave, array, array2, num, num2)

    def SPR4E(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 5)
        array2 = self.w_adj_ws(liS, liE, -1)
        num = len(array2) - 1
        num2 = len(array) - 1
        if num2 >= 3:
            num3 = self.w_rr(array[3], array[2], array[1])
            num4 = array[1] - array[0]
            num5 = array[2] - array[1]
            num6 = array[3] - array[2]
            if num3 > 2.618:
                if num6 <= num4 and self.w_steep_ret(array[1], array[2]):
                    elwave.AddSP("※ There is a strong possibility of a missing x-wave hidden toward the middle of m0", 256)
                if (num2 >= 5 and num >= 1 and not self.w_byond(array2[1], array2[0], array[3])
                        and not self.w_byond(array2[1], array2[0], array[5])
                        and self.w_rr(array[4], array[3], array[2]) >= 0.618):
                    elwave.AddSP("※ m1 may be the first leg of an Elongated Flat", 8)
            elif num3 >= 1.0:
                if num2 >= 4:
                    num7 = self.w_rr(array[4], array[3], array[2])
                    if self.w_steep_ret(array[2], array[3]) or (num7 > 1.0 and array[4] - array[3] <= num6):
                        elwave.AddSP("※ :F3 is the only reasonable choice", 8)
                    if num3 <= 1.618 and num7 < 1.0 and self.w_steep_ret(array[3], array[4]):
                        elwave.AddSP("※ m1 could be the x-wave of a Complex Correction", 256)
                        if self.w_l_v(array2[1], array2[0]) >= 0.618 * self.w_l_v(array2[0], array[0]):
                            elwave.AddSP("※ m0 would be missing an x-wave toward its center", 256)
                    if num >= 1:
                        if self.w_steep_ret(array[1], array[2]):
                            if self.w_rr(array2[1], array2[0], array[0]) <= 0.618 and num7 <= 0.618:
                                if _div(self._plist[array[3]] - self._plist[array[2]], self._plist[array[1]] - self._plist[array[0]]) >= 1.0 and num6 <= num4:
                                    elwave.AddSP("※ m1 may be part of a Complex Correction where m0 contains a missing x-wave toward its center, or m1 is the x-wave following a Zigzag", 8)
                                    elwave.AddSP("※ m1 may be part of a Complex Correction where m0 contains a missing x-wave toward its center, or m1 is the x-wave following a Zigzag", 4112)
                                    if elwave.Linked and liS in self._mw_list:
                                        self._mw_list[liS].AddSP("", 512)
                                elif num2 >= 5:
                                    num8 = array[5] - array[2]
                                    if _div(self._plist[array[5]] - self._plist[array[2]], self._plist[array[1]] - self._plist[array[0]]) >= 1.0 and num8 <= num4:
                                        elwave.AddSP("※ m1 may be part of a Complex Correction where m0 contains a missing x-wave toward its center, or m1 is the x-wave following a Zigzag", 8)
                                        elwave.AddSP("※ m1 may be part of a Complex Correction where m0 contains a missing x-wave toward its center, or m1 is the x-wave following a Zigzag", 4112)
                                        if elwave.Linked and liS in self._mw_list:
                                            self._mw_list[liS].AddSP("", 512)
                        elif self.w_total_ret(array[1], array[2]) > num5 and self.w_rr(array2[1], array2[0], array[0]) <= 0.618 and num7 <= 0.618:
                            if _div(self._plist[array[3]] - self._plist[array[2]], 1.618 * (self._plist[array[1]] - self._plist[array[0]])) >= 1.0 and num6 <= num4:
                                elwave.AddSP("※ m1 may be part of a Complex Correction where m0 contains a missing x-wave toward its center", 8)
                                elwave.AddSP("※ m1 may be part of a Complex Correction where m0 contains a missing x-wave toward its center", 4112)
                                if elwave.Linked and liS in self._mw_list:
                                    self._mw_list[liS].AddSP("", 512)
                            elif num2 >= 5:
                                num9 = array[5] - array[2]
                                if _div(self._plist[array[5]] - self._plist[array[2]], 1.618 * (self._plist[array[1]] - self._plist[array[0]])) >= 1.0 and num9 <= num4:
                                    elwave.AddSP("※ m1 may be part of a Complex Correction where m0 contains a missing x-wave toward its center", 8)
                                    elwave.AddSP("※ m1 may be part of a Complex Correction where m0 contains a missing x-wave toward its center", 4112)
                                    if elwave.Linked and liS in self._mw_list:
                                        self._mw_list[liS].AddSP("", 512)
                if self.w_total_ret(array[1], array[2]) > array[2] - array[1]:
                    elwave.AddSP("※ A Flat or Triangle is probably involved", 8)
                if num >= 1:
                    if self.w_mw_cnt(array2[0], array[0]) >= 3 or self.w_rr(array2[1], array2[0], array[0]) <= 0.618:
                        elwave.AddSP("※ m1 may be an x-wave of a Complex Correction", 256)
                    if self.w_l_v(array2[1], array2[0]) <= 0.618 * self.w_l_v(array2[0], array[0]):
                        elwave.AddSP("※ m1 may be an x-wave of a Complex Correction", 256)
        return self._tail(elwave, array, array2, num, num2)

    def SPR5A(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -4)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        if num2 >= 2:
            num5 = self.w_l_v(array[1], array[2])
            num6 = array[2] - array[1]
            if self.w_mw_cnt(array[1], array[2]) > 3:
                num7 = self.w_r_3w(array[0], array[1], array[2])
                if num7 < 0.618:
                    elwave.AddSP("※ m1 may be 3 of a 5th Failure Impulse pattern", 1)
                    elwave.AddSP("※ A complex correction may be unfolding with the first wave of m2 representing an x-wave", 2)
                    elwave.AddSP("※ m1 contains a missing x-wave or missing b-wave in its center", 2)
                    if num7 >= 0.25:
                        elwave.AddSP("", 8)
                    if elwave.Linked and liE in self._mw_list:
                        self._mw_list[liE].AddSP("", 512)
                else:
                    elwave.AddSP("※ m1 may have completed wave-a of a Flat with a complex b-wave", 8)
                    elwave.AddSP("※ m1 may be 3 of a 5th failure impulse pattern", 1)
            else:
                if self.w_steep_ret(array[0], array[1]):
                    if num >= 2:
                        num8 = self.w_l_v(array2[1], array2[0])
                        num9 = self.w_l_v(array2[2], array2[1])
                        flag = self.w_similar(array2[2], array2[1], array2[0], array[0])
                        flag2 = self.w_overlap(array2[2], array2[1], array2[0], array[0])
                        if num5 < num9:
                            elwave.AddSP("※ A Flat or Zigzag may have concluded with m1", 4)
                        if num >= 3:
                            num10 = self.w_l_v(array2[3], array2[2])
                            if not flag2 and num5 > num9 and not flag and self.w_impulse_rule_3(num10, num8, num4):
                                elwave.AddSP("※ A Trending Impulse pattern may have concluded with m1", 4)
                            if num2 >= 3:
                                num11 = self.w_rr(array[3], array[2], array[1])
                                num12 = array[1] - array2[3]
                                num13 = self.w_rt_over(array2[3], array2[2], array[1])
                                num14 = self.w_rt_over(array[1], array[2], array[2])
                                if (num9 < num8 and self.w_impulse_rule_3(num10, num8, num4) and num11 < 0.618
                                        and 0 < num13 and num13 <= num12 / 2.0
                                        and (num14 == 0 or num14 + num6 > num12 * 4) and num2 >= 4
                                        and num4 < max(num5, self.w_l_v(array[1], array[4])) / 2.0
                                        and self._mw_list[array2[0]].SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE):
                                    elwave.AddSP("※ A Terminal Impulse may have completed at the end of m1", 64)
                            if num >= 4:
                                num15 = self.w_l_v(array2[4], array2[3])
                                if num5 > num9 and num15 > num10:
                                    elwave.AddSP("※ A Zigzag or Flat may have concluded with m1", 4)
                                if num5 > num9 and num10 > num9 and num15 < num10:
                                    elwave.AddSP("※ m1 may be the end of a Standard Elliott pattern which is part of a Complex Correction where mn2 is an x-wave", 4)
                                    if elwave.Linked and array2[1] in self._mw_list:
                                        self._mw_list[array2[1]].AddSP("", 256)
                                    if num8 >= 1.618 * num3:
                                        elwave.AddSP("※ the Standard pattern unfolding is probably a Zigzag", 4)
                                    elif num8 >= num3:
                                        elwave.AddSP("※ the Standard pattern unfolding is probably a Flat", 4)
                    if num2 >= 3:
                        num16 = self.w_l_v(array[2], array[3])
                        if num16 > 0.618 * num4 and num16 < num4:
                            elwave.AddSP("※ m1 may be part of an Irregular Failure Flat", 8)
                        if num16 >= num5 and num3 >= 1.618 * num5 and self.w_steep_ret(array[2], array[3]):
                            elwave.AddSP("※ m1 may be part of an Irregular Flat", 8)
                else:
                    if num2 >= 3 and num >= 1:
                        num16 = self.w_l_v(array[2], array[3])
                        if num5 < 0.618 * max(num4, self.w_l_v(array2[1], array[1])):
                            if num16 < num5:
                                elwave.AddSP("※ The market may be concluding a Complex Correction in which m1 will be the most extreme price obtained by the market for a period at least twice that of the combined times of m0-m2", 8)
                            else:
                                elwave.AddSP("※ The market could be forming a Complex Correction or an Expanding Triangle", 8, 16, 4)
                        if num16 < num5 and self.w_l_v(array2[1], array2[0]) >= 0.618 * num4 and self.w_steep_ret(array[2], array[3]):
                            elwave.AddSP("※ m1 may be part of a Flat which concludes a larger pattern", 8)
                    if num >= 2 and num5 < self.w_l_v(array2[2], array2[1]):
                        elwave.AddSP("※ A Zigzag, which is part of a Contracting Triangle, may have concluded with m1", 4)
                if num2 >= 4:
                    num16 = self.w_l_v(array[2], array[3])
                    if num16 > num5 and self.w_l_v(array[3], array[4]) > num16:
                        if num3 < 0.618 * num4:
                            elwave.AddSP("※ m1 may have begun an Expanding Triangle", 2056)
                        elif num3 >= 0.618 * num4 and num3 <= num4:
                            elwave.AddSP("※ The market may be forming an Expanding Triangle", 2064)
        return self._tail(elwave, array, array2, num, num2)

    def SPR5B(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        if num2 >= 2:
            num5 = self.w_l_v(array[1], array[2])
            if num2 >= 3:
                num6 = self.w_l_v(array[2], array[3])
                num7 = self.w_rr(array[3], array[2], array[1])
                if num7 > 1.0:
                    if num3 - num4 < 1.618 * num4 - num3:
                        if num >= 1:
                            num8 = self.w_l_v(array2[1], array2[0])
                            if num8 > num3:
                                elwave.RemoveSP(16)
                                elwave.AddSP("※ m1 may be the wave-b of a Flat", 16, 128)
                            elif num8 < num3:
                                elwave.RemoveSP(16)
                                elwave.AddSP("※ m1 may be the x-wave of a Complex Correction", 256)
                    else:
                        elwave.AddSP("", 8)
                        if num >= 1:
                            num8 = self.w_l_v(array2[1], array2[0])
                            if num8 > num3:
                                elwave.AddSP("※ m2 is probably the end of a Zigzag", 128)
                            elif num8 < num3:
                                elwave.AddSP("※ m1 may be an x-wave of a Complex Correction which ends with m4", 256)
                else:
                    if num6 >= 0.618 * num4:
                        elwave.AddSP("", 8)
                    if num7 <= 0.618 and num2 >= 4:
                        num9 = self.w_l_v(array[3], array[4])
                        flag = self.w_steep_ret(array[0], array[1])
                        if (self.w_overlap(array[0], array[1], array[2], array[3]) and num9 <= 2.618 * num5
                                and self.w_impulse_rule_3(num3, num5, num9) and self.w_steep_ret(array[3], array[4])):
                            num10 = array[4] - array2[0]
                            num11 = self.w_rt_over(array2[0], array[0], array[4])
                            if 0 < num11 and num11 <= num10 / 2.0:
                                elwave.AddSP("※ The market possibly completed a Terminal pattern at m4", 16)
                                elwave.RemoveSP(8)
                                elwave.AddSP("", 4104)
                        if num >= 1:
                            num8 = self.w_l_v(array2[1], array2[0])
                            if flag and self.w_almost_same(num5, 1.618 * num4) and num4 <= 1.618 * num8 and num5 - num6 + num9 > num3:
                                elwave.AddSP("※ m1 may have completed a Flat", 4)
                            if num >= 3:
                                num12 = self.w_l_v(array2[2], array2[1])
                                num13 = self.w_l_v(array2[3], array2[2])
                                if (flag and self.w_almost_same(num5, 1.618 * num4) and num8 >= 0.618 * num3
                                        and num12 >= 0.618 * num8 and num12 <= 1.618 * num8
                                        and num13 >= 0.618 * num12 and num13 <= 1.618 * num12
                                        and num5 - num6 + num9 > num3):
                                    elwave.AddSP("※ m1 may have completed a Contracting Triangle", 64)
                if num6 >= 0.618 * num4 and self.w_almost_same(num5, 0.618 * num3):
                    elwave.AddSP("※ m1 may be the first leg of an Irregular Failure Flat", 8)
            if (num >= 2 and self.w_l_v(array2[1], array2[0]) < num3 and self.w_steep_ret(array[1], array[2])
                    and self.w_impulse_rule_3(self.w_l_v(array2[2], array2[1]), num3, num5)):
                num14 = array[2] - array2[2]
                num15 = self.w_rt_over(array2[2], array2[1], array[2])
                if 0 < num15 and num15 <= num14 / 2.0:
                    elwave.AddSP("※ A Terminal Impulse pattern could have completed at the end of m2", 2080)
        return self._tail(elwave, array, array2, num, num2)

    def SPR5C(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 6)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        if num2 >= 2:
            num5 = self.w_l_v(array[1], array[2])
            if num2 >= 3:
                num6 = self.w_l_v(array[2], array[3])
                num7 = self.w_rr(array[3], array[2], array[1])
                if num7 > 1.0:
                    if num7 <= 1.618:
                        elwave.AddSP("※ There is the unkikely possibility an Expanding Triangle is forming", 16)
                    if (num >= 2 and self.w_l_v(array2[1], array2[0]) < num3 and self.w_steep_ret(array[1], array[2])
                            and self.w_impulse_rule_3(self.w_l_v(array2[2], array2[1]), num3, num5)
                            and self.w_overlap(array2[1], array2[0], array[0], array[1])):
                        num8 = array[2] - array2[2]
                        num9 = self.w_rt_over(array2[2], array2[1], array[2])
                        if 0 < num9 and num9 <= num8 / 2.0:
                            elwave.AddSP("※ A Terminal Impulse pattern may be concluded with m2", 2080)
                elif num7 < 0.618:
                    if num2 >= 4:
                        if (self.w_overlap(array[0], array[1], array[2], array[3])
                                and self.w_l_v(array[3], array[4]) < num5 and self.w_steep_ret(array[3], array[4])):
                            num10 = array[4] - array2[0]
                            num11 = self.w_rt_over(array2[0], array[0], array[4])
                            if 0 < num11 and num11 <= num10 / 2.0:
                                elwave.AddSP("※ The market possibly completed a Terminal pattern at m4", 16)
                        if (num >= 1 and self.w_steep_ret(array[0], array[1]) and self.w_almost_same(num5, 1.618 * num4)
                                and self.w_almost_same(num4, 0.618 * num3) and num4 <= 1.618 * self.w_l_v(array2[1], array2[0])
                                and num5 - self.w_l_v(array[2], array[3]) + self.w_l_v(array[3], array[4]) > num3):
                            elwave.AddSP("※ m1 may have completed a Flat", 4)
                    if num >= 3:
                        num12 = self.w_l_v(array2[1], array2[0])
                        num13 = self.w_l_v(array2[2], array2[1])
                        num14 = self.w_l_v(array2[3], array2[2])
                        if (self.w_steep_ret(array[0], array[1]) and self.w_almost_same(num5, 1.618 * num4)
                                and num12 >= 0.618 * num3 and num13 >= 0.618 * num12 and num13 <= 1.618 * num12
                                and num14 >= 0.618 * num13 and num14 <= 1.618 * num13 and num2 >= 4
                                and num5 - self.w_l_v(array[2], array[3]) + self.w_l_v(array[3], array[4]) > num3):
                            elwave.AddSP("※ m1 may have completed a Contracting Triangle", 64)
                if num2 >= 6:
                    num15 = self.w_l_v(array[3], array[4])
                    if (num6 >= 0.618 * num4 and num6 <= 1.618 * num4 and num5 < 0.618 * num3 and num15 >= num5
                            and (num15 >= num3 or num15 - self.w_l_v(array[4], array[5]) + self.w_l_v(array[5], array[6]) >= num3)):
                        elwave.AddSP("※ m1 may be the first leg of an Irregular Flat (either variation) or a Running Triangle", 8)
        return self._tail(elwave, array, array2, num, num2)

    def SPR5D(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 6)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        num5 = self.w_l_v(array[1], array[2])
        if self.w_mw_cnt(array[1], array[2]) >= 3:
            elwave.AddSP("", 8)
        if num2 >= 3:
            num6 = self.w_l_v(array[2], array[3])
            num7 = self.w_rr(array[3], array[2], array[1])
            if num7 > 1.0:
                if num >= 2 and self.w_steep_ret(array[1], array[2]) and self.w_l_v(array2[2], array2[1]) < num3:
                    num8 = array[2] - array2[2]
                    num9 = self.w_rt_over(array2[2], array2[1], array[2])
                    if 0 < num9 and num9 <= num8 / 2.0:
                        elwave.AddSP("※ There is the outside possibility of a 3rd Extension Terminal pattern completing with m2", 2080)
            elif num2 >= 4:
                num10 = self.w_l_v(array[3], array[4])
                if num7 > 0.618:
                    if num10 < 0.618 * num3 and array[1] - array[0] < array[0] - array2[0]:
                        elwave.AddSP("※ m1 may be an x-wave within a Complex Correction", 256)
                    if num10 >= 0.618 * num3:
                        elwave.AddSP("", 8)
                else:
                    if num7 < 0.618:
                        elwave.AddSP("", 8, 16)
                    if self.w_overlap(array[0], array[1], array[2], array[3]) and num10 < num5 and self.w_steep_ret(array[3], array[4]):
                        num11 = array[4] - array2[0]
                        num12 = self.w_rt_over(array2[0], array[0], array[4])
                        if 0 < num12 and num12 <= num11 / 2.0:
                            elwave.AddSP("※ The market possibly completed a Terminal pattern at m4", 16)
                    if num >= 1:
                        num13 = self.w_l_v(array2[1], array2[0])
                        if num13 >= 0.618 * num3 and self.w_steeper(array[1], array[4], array2[0], array[0]):
                            elwave.AddSP("※ There is a remote chance m1 completed a Contracting Triangle", 2112)
                        num14 = array2[0] - array2[1]
                        num15 = array[0] - array2[0]
                        num16 = liE - liS
                        if (self.w_almost_same(num13, num3) and num16 >= num14 and num15 > num14 and num15 > num16
                                and self.w_steeper(array[1], array[4], array2[0], array[0])):
                            elwave.AddSP("※ There is a remote chance m1 completed a severe c-Failure Flat", 4100)
                if (num2 >= 6 and num6 >= 0.618 * num4 and num6 < num5 and num10 >= num5
                        and (num10 >= 0.618 * num3 or (self.w_rr(array[5], array[4], array[3]) <= 1.0
                             and num10 - self.w_l_v(array[4], array[5]) + self.w_l_v(array[5], array[6]) >= 0.618 * num3))):
                    elwave.AddSP("※ There is a good chance m1 is the first leg of an Irregular Failure Flat", 8)
        return self._tail(elwave, array, array2, num, num2)

    def SPR6A(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 3)
        array2 = self.w_adj_ws(liS, liE, -3)
        num = len(array2) - 1
        num2 = len(array) - 1
        self.w_l_v(array2[0], array[0])
        num3 = self.w_l_v(liS, liE)
        if num2 >= 2:
            num4 = self.w_l_v(array[1], array[2])
            if self.w_mw_cnt(array[1], array[2]) > 3:
                if self.w_r_3w(array[0], array[1], array[2]) < 0.618:
                    elwave.AddSP("※ First wave of m2 may be an x-wave, or m1 may be hiding an x-wave", 2)
                    elwave.AddSP("※ m1 may be 3 of a 5th Failure Impulse pattern", 1)
                    if elwave.Linked and liE in self._mw_list:
                        self._mw_list[liE].AddSP("", 512)
                else:
                    elwave.AddSP("※ m1 may have completed wave-a of a Flat with a complex b-wave", 8)
                    elwave.AddSP("※ m1 may have completed wave-3 of a 5th Failure Impulse pattern", 1)
            else:
                if num2 >= 3:
                    if self.w_rr(array[3], array[2], array[1]) <= 0.618:
                        elwave.AddSP("", 4)
                        if num >= 2 and self.w_overlap(array2[2], array2[1], array2[0], array[0]):
                            elwave.AddSP("", 64)
                    else:
                        elwave.AddSP("", 4)
                if self.w_steep_ret(array[0], array[1]):
                    elwave.AddSP("※ A Trending Impulse pattern may have completed with m1", 4)
                    if num2 >= 3 and num >= 3:
                        if self.w_l_v(array[2], array[3]) < num4 and self.w_overlap(array2[2], array2[1], array2[0], array[0]):
                            num5 = array[1] - array2[3]
                            num6 = self.w_rt_over(array2[3], array2[2], array[1])
                            if 0 < num6 and num6 <= num5 / 2.0:
                                elwave.AddSP("※ m1 may have completed a Terminal Impulse pattern", 2112)
                        num7 = self.w_l_v(array[2], array[3])
                        if num7 > 0.618 * num4 and num7 < num4 and elwave.SP(EW_SP.LastThree, strict=False) > EW_SP.NONE:
                            elwave.AddSP("※ m2 is probably an x-wave or the Terminal pattern which completed with m1 is within a larger Triangle", 64)
                            if elwave.Linked and array[2] in self._mw_list:
                                self._mw_list[array[2]].AddSP("", 256)
                elif num >= 2:
                    num8 = self.w_l_v(array2[1], array2[0])
                    num9 = self.w_l_v(array2[2], array2[1])
                    if not self.w_byond(array2[2], array2[1], array[2]) and num8 >= 0.618 * num3 and num9 < num8:
                        elwave.AddSP("※ m1 may be wave-a of a Flat pattern which concludes a Complex Correction and m0 is the x-wave of the pattern", 8)
                        if elwave.Linked and array[0] in self._mw_list:
                            self._mw_list[array[0]].AddSP("", 256)
        return self._tail(elwave, array, array2, num, num2)

    def SPR6B(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 3)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        if num2 >= 3:
            num5 = self.w_l_v(array[1], array[2])
            num6 = self.w_l_v(array[2], array[3])
            if self.w_mw_cnt(array[2], array[3]) > 3:
                if self.w_r_3w(array[1], array[2], array[3]) < 0.618:
                    elwave.AddSP("※ First wave of m3 may be a x-wave, or m2 may be hiding a x-wave, or m2 may be 3 of a 5th Failure Impulse pattern", 1, 2)
                    if self._mw_list[array[1]].mHA.find(":5? ○ :F3/x:c3? ") < 0 and elwave.Linked and array[2] in self._mw_list:
                        self._mw_list[array[2]].AddSP("", 512)
                else:
                    elwave.AddSP("※ m2 may have completed wave-a of a Flat with a complex b-wave", 8)
                    elwave.AddSP("※ m1 may have completed 3 of a 5th Failure Impulse pattern", 1)
            else:
                num7 = array[0] - array2[0]
                num8 = liE - liS
                num9 = array[2] - array[1]
                num10 = array[3] - array[2]
                if num >= 1:
                    num11 = array2[0] - array2[1]
                    num12 = self.w_l_v(array2[1], array2[0])
                    if self.w_steep_ret(array[0], array[1]):
                        num13 = self.w_rr(array[3], array[2], array[1])
                        if ((self.w_steep_ret(array[1], array[2]) or (num13 <= 0.618 and num10 <= num9))
                                and num12 >= 0.618 * num4 and num11 >= 0.618 * num8):
                            elwave.AddSP("※ There is a chance m1 completed the c-wave of a Flat", 4)
                        if num13 < 0.618 and num12 >= 0.618 * num3 and num11 >= 0.618 * num7:
                            elwave.AddSP("※ There is a chance m1 completed a Contracting Triangle", 64)
                            elwave.AddSP("※ There is a chance m1 completed one of several Flat variations", 4)
                    elif self.w_mw_cnt(array[1], array[2]) >= 3 and num5 >= num12 and num5 >= num3:
                        elwave.AddSP("※ m1 may be one of the middle legs of a Triangle", 16)
                    if num >= 2:
                        num14 = self.w_l_v(array2[2], array2[1])
                        if num8 <= num7 and num8 <= num9 and num14 < num12:
                            elwave.AddSP("※ m1 may be the x-wave of a Complex Correction", 256)
                        if self.w_steep_ret(array[1], array[2]) and self.w_impulse_rule_3(num14, num3, num5):
                            num15 = array[2] - array2[2]
                            num16 = self.w_rt_over(array2[2], array2[1], array[2])
                            if 0 < num16 and num16 <= num15 / 2.0:
                                elwave.AddSP("※ A Terminal pattern may have concluded with m2", 32)
                if num8 >= num7 or num8 >= num9 or self.w_almost_same(num3, 1.618 * num4):
                    elwave.AddSP("※ m1 may be part of a Zigzag or Impulse pattern", 8)
                if num6 > num5 and num6 < 1.618 * num5:
                    elwave.RemoveSP(8)
                    elwave.AddSP("※ An Expanding Triangle may be forming", 4104)
        return self._tail(elwave, array, array2, num, num2)

    def SPR6C(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        elwave.AddSP("※ Despite any particular circumstances, :F3 is a good possibility", 8)
        array = self.w_adj_ws(liS, liE, 3)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(array[1], array[2])
        if num2 >= 3 and num >= 1 and self.w_steep_ret(array[0], array[1]):
            num5 = self.w_rr(array[3], array[2], liE)
            num6 = self.w_l_v(array2[1], array2[0])
            num7 = array[0] - array2[0]
            num8 = array[2] - array[1]
            if (num5 < 0.618 and num4 > num3 and num8 <= num7 and num6 > 0.618 * num3 and num6 < 1.618 * num3
                    and self.w_steeper(array[1], array[2], array2[0], array[0])):
                elwave.AddSP("※ m1 may have completed a Contracting Triangle", 64)
                elwave.AddSP("※ m1 may have completed a severe C-Failure Flat", 2052)
        if num >= 2:
            flag = self.w_overlap(array2[1], array2[0], array[0], array[1])
            if self.w_steep_ret(array[1], array[2]) and flag and self.w_impulse_rule_3(self.w_l_v(array2[2], array2[1]), num3, num4):
                num9 = array[2] - array2[2]
                num10 = self.w_rt_over(array2[2], array2[1], array[2])
                if 0 < num10 and num10 <= num9 / 2.0:
                    elwave.AddSP("※ A Terminal pattern may have concluded with m2", 32)
        if num2 >= 3:
            num11 = self.w_l_v(array[2], array[3])
            if num11 > num4 and num11 < 1.618 * num4:
                elwave.AddSP("※ There is the unlikely possibility that an Expanding Triangle is forming", 2064)
        return self._tail(elwave, array, array2, num, num2)

    def SPR6D(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        len_ = self.w_l_v(array2[0], array[0])
        len2 = self.w_l_v(array[1], array[2])
        num3 = array[0] - array2[0]
        num4 = liE - liS
        num5 = array[2] - array[1]
        if (num3 - 1 <= num4 or num5 - 1 <= num4) and num4 >= min(num3, num5):
            elwave.AddSP("※ m1 is either the first phase of a larger correction or the completion of a correction within a Zigzag or Impulse pattern", 8)
        if num2 >= 4:
            num6 = self.w_rr(array[3], array[2], array[1])
            num7 = array[4] - array[1]
            if num6 < 0.618 and num7 <= num3 and self.w_steeper(array[1], array[4], array2[0], array[0]):
                elwave.AddSP("※ There is a small chance that m1 completed a Contracting Triangle", 2112)
                elwave.AddSP("※ There is a small chance that m1 completed a a severe C-Failure Flat", 4100)
        if num >= 2:
            flag = self.w_overlap(array2[1], array2[0], array[0], array[1])
            if self.w_steep_ret(array[1], array[2]) and flag and self.w_impulse_rule_3(self.w_l_v(array2[2], array2[1]), len_, len2):
                num8 = array[2] - array2[2]
                num9 = self.w_rt_over(array2[2], array2[1], array[2])
                if 0 < num9 and num9 <= num8 / 2.0:
                    elwave.AddSP("※ A Terminal pattern may have concluded with m2", 32)
        return self._tail(elwave, array, array2, num, num2)

    def SPR7A(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 3)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        if num2 >= 2:
            if self.w_mw_cnt(array[1], array[2]) > 3:
                if self.w_r_3w(array[0], array[1], array[2]) < 0.618:
                    elwave.AddSP("※ First wave of m2 may be an x-wave, or m1 may be hiding an x-wave", 2)
                    elwave.AddSP("※ m1 may be 3 of a 5th Failure Impulse pattern", 1)
                    if elwave.Linked and array[2] in self._mw_list:
                        self._mw_list[array[2]].AddSP("", 256)
                else:
                    elwave.AddSP("※ m1 may have completed wave-a of a Flat with a complex b-wave", 8)
                    elwave.AddSP("※ m1 may have completed wave-3 of a 5th Failure Impulse pattern", 1)
            else:
                elwave.AddSP("※ No matter what the surrounding evidence, an :L5 is very likely", 4)
                if num2 >= 3 and num >= 2:
                    num3 = self.w_rr(array[3], array[2], array[1])
                    flag = self.w_overlap(array2[2], array2[1], array2[0], array[0])
                    if (num3 < 0.618 and self.w_l_v(array2[2], array2[1]) < self.w_l_v(array2[1], array2[0])
                            and flag):
                        elwave.AddSP("※ There is the possibility a Terminal Impulse pattern completed with m1", 2112)
        return self._tail(elwave, array, array2, num, num2)

    def SPR7B(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        if num2 >= 3:
            num5 = self.w_l_v(array[1], array[2])
            num6 = self.w_l_v(array[2], array[3])
            if self.w_mw_cnt(array[2], array[3]) > 3:
                if self.w_r_3w(array[1], array[2], array[3]) < 0.618:
                    elwave.AddSP("※ First wave of m3 may be an x-wave, or m2 may be hiding a wave-b, or m2 may be 3 of a 5th Failure Impulse pattern", 8, 16, 64, 4)
                    elwave.AddSP("※ If 5 waves are needed to retrace m2 by 61.8% and more, exclude :F3 and '3 in 5x' case", 8, 16, 64, 4)
                    if elwave.Linked:
                        if array[2] in self._mw_list:
                            self._mw_list[array[2]].AddSP("", 512)
                        if array[3] in self._mw_list:
                            self._mw_list[array[3]].AddSP("", 512)
                else:
                    elwave.AddSP("※ m2 may have completed wave-a of a Flat with a complex wave-b", 8)
                    elwave.AddSP("※ m1 may have completed wave-3 of a 5th Failure Impulse pattern", 1)
            else:
                num7 = self.w_rr(array[3], array[2], array[1])
                if num3 >= 0.618 * num4 and num6 > num5 and num6 < 2.618 * num5:
                    elwave.AddSP("※ m1 may be part of an Expanding Triangle", 16)
                    if num2 >= 4 and self.w_l_v(array[3], array[4]) >= 0.618 * num6:
                        elwave.AddSP("", 8)
                if (num4 <= 0.618 * num3
                        and (self.w_steep_ret(array[1], array[2])
                             or (num7 <= 0.618 and array[3] - array[2] <= array[2] - array[1]))
                        and self.w_steeper(array[1], array[2], array2[0], array[0])):
                    elwave.AddSP("※ There is a good chance m1 completed a Contracting Triangle", 64)
                    elwave.AddSP("※ There is a good chance m1 completed a c-Failure Flat", 4)
                if num7 >= 0.618 and num7 < 1.0:
                    elwave.AddSP("※ An Elongated Flat is the most likely pattern which concluded with m2", 16)
                if self.w_steep_ret(array[1], array[2]):
                    elwave.AddSP("※ A Trending Impulse pattern may have concluded with m2", 4)
                    if (num >= 2 and self.w_l_v(array2[1], array2[0]) < num3
                            and self.w_impulse_rule_3(self.w_l_v(array2[2], array2[1]), num3, num5)):
                        num8 = array[2] - array2[2]
                        num9 = self.w_rt_over(array2[2], array2[1], array[2])
                        if 0 < num9 and num9 <= num8 / 2.0:
                            elwave.AddSP("※ A 5th Extension Terminal may have completed with m2", 32)
        return self._tail(elwave, array, array2, num, num2)

    def SPR7C(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 3)
        array2 = self.w_adj_ws(liS, liE, -4)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = array[0] - array2[0]
        num5 = liE - liS
        if num2 >= 2:
            num6 = self.w_l_v(array[1], array[2])
            num7 = array[2] - array[1]
            if num4 <= num5 or num7 <= num5:
                elwave.AddSP("※ Despite any other circumstances, :F3 is a good possibility", 8)
            if num >= 2:
                num8 = self.w_l_v(array2[2], array2[1])
                if self.w_steep_ret(array[1], array[2]) and num3 > num8:
                    num9 = array[2] - array2[2]
                    num10 = self.w_rt_over(array2[2], array2[1], array[2])
                    if 0 < num10 and num10 <= num9 / 2.0:
                        elwave.AddSP("※ An Expanding Terminal Impulse pattern may have concluded with m2", 32)
                if (self.w_steep_ret(array[0], array[1]) and num6 >= 1.618 * num3
                        and self.w_tr_end(array2[2], array2[0], array[0], array[1])):
                    elwave.AddSP("※ A Running Correction could have completed with m1", 4)
                if self.w_steeper(array[1], array[2], array2[0], array[0]):
                    if num >= 4 and self.w_l_v(array2[4], array2[3]) > num8:
                        elwave.AddSP("※ m1 may have completed a Contracting Triangle", 64)
                    if (num8 >= 1.618 * num3 and num8 >= 0.618 * num6
                            and self._mw_list[array2[0]].SP(EW_SP.FirstThree, strict=False) > EW_SP.NONE):
                        elwave.AddSP("※ m1 may have completed an Irregular Failure Flat", 4)
        return self._tail(elwave, array, array2, num, num2)

    def SPR7D(self, elwave):
        liS = elwave.liS
        liE = elwave.liE
        array = self.w_adj_ws(liS, liE, 4)
        array2 = self.w_adj_ws(liS, liE, -2)
        num = len(array2) - 1
        num2 = len(array) - 1
        num3 = self.w_l_v(array2[0], array[0])
        num4 = self.w_l_v(liS, liE)
        num5 = array[0] - array2[0]
        num6 = liE - liS
        if num2 >= 2:
            num7 = self.w_l_v(array[1], array[2])
            num8 = array[2] - array[1]
            if (num5 - 1 <= num6 or num8 - 1 <= num6) and num6 >= min(num5, num8):
                elwave.AddSP("※ m1 may be part of a Zigzag or Impulse pattern", 8)
            if num6 <= num5 or num6 <= num8:
                elwave.AddSP("", 16)
                if num2 >= 4 and num >= 1:
                    num9 = self.w_l_v(array[2], array[3])
                    num10 = self.w_l_v(array[3], array[4])
                    num11 = self.w_l_v(array2[1], array2[0])
                    if (num3 >= num11 and num3 <= 1.618 * num11 and num7 <= 1.618 * num3
                            and num10 >= 0.382 * num7 and num9 > num7 and num10 < num9):
                        elwave.AddSP("※ m1 may be the x-wave of a Complex correction which starts with a Flat and ends with a Flat or Triangle", 256)
                    if num >= 2:
                        num12 = self.w_l_v(array2[2], array2[1])
                        num13 = num12 - num11 + num3
                        if (num12 >= 1.618 * num11 and num11 < num3 and num4 < 0.618 * num13
                                and num9 > num7 and num10 < num9):
                            elwave.AddSP("※ m1 may be the x-wave of a Double Zigzag or a Complex correction which begins with a Zigzag", 256)
                if num >= 1:
                    num11 = self.w_l_v(array2[1], array2[0])
                    flag = self.w_t_similar(array2[1], array2[0], array[0], array[1])
                    flag2 = self.w_p_similar(array2[1], array2[0], array[0], array[1])
                    flag3 = self.w_overlap(array2[1], array2[0], array[0], array[1])
                    if (flag or flag2) and num11 < num3 and num >= 2:
                        num12 = self.w_l_v(array2[2], array2[1])
                        if self.w_impulse_rule_3(num12, num3, num7) and not self.w_impulse_rule_l(num12, num3, num7):
                            elwave.RemoveSP(16)
                            elwave.AddSP("※ m1 may be part of a Complex Double Zigzag", 256)
                            if num3 != max(num12, max(num3, num7)):
                                elwave.AddSP("※ But if there are other Structure label possibilities for m1 other than :c3, x could be at the end of mn1 or m3", 256)
                            else:
                                if elwave.Linked and array2[0] in self._mw_list:
                                    self._mw_list[array2[0]].AddSP("", 512)
                                elwave.AddSP("※ In that case, the pattern must be retraced by 61,8 ~ 100%", 256)
                                elwave.AddSP("※ If not, there may be other pattern preceding", 256)
            if num >= 1:
                num11 = self.w_l_v(array2[1], array2[0])
                flag = self.w_t_similar(array2[1], array2[0], array[0], array[1])
                flag2 = self.w_p_similar(array2[1], array2[0], array[0], array[1])
                flag3 = self.w_overlap(array2[1], array2[0], array[0], array[1])
                if (flag or flag2) and self.w_steep_ret(array[0], array[1]) and num7 >= 1.618 * num3:
                    if not flag3 and not self.w_steep_ret(array[1], array[2]):
                        elwave.AddSP("※ m1 may have completed a Running Correction", 4)
                    if num2 >= 3 and self.w_rr(array[3], array[2], array[1]) <= 0.618 and num >= 2:
                        num12 = self.w_l_v(array2[2], array2[1])
                        if num12 < 1.618 * num3 and elwave.SP(EW_SP.LastFive, strict=False) > EW_SP.NONE:
                            elwave.AddSP("※ m1 simultaneously terminates more than one Elliott pattern, each of consecutively larger degree", 4)
                if num2 >= 3:
                    if self.w_rr(array[3], array[2], array[1]) <= 0.618:
                        if self.w_steeper(array[1], array[2], array2[0], array[0]) and num11 <= 1.618 * num3:
                            if flag3 and (self._mw_list[array[0]].SP(EW_SP.CenterThree, strict=False) > EW_SP.NONE
                                          or self._mw_list[array[0]].SP(EW_SP.SecondToLastThree, strict=False) > EW_SP.NONE):
                                elwave.AddSP("※ There is small chance m1 completed a Contracting Triangle", 2112)
                            if (flag or flag2) and flag3:
                                elwave.AddSP("※ m1 may have completed an Irregular or c-Failure Flat", 4)
                        if num7 >= 0.618 * num3 and num7 <= 1.618 * num3 and num11 < num3:
                            elwave.AddSP("※ m1 may be an x-wave of a Complex Corrective pattern", 256)
                    if num2 >= 4 and num >= 2:
                        num12 = self.w_l_v(array2[2], array2[1])
                        if (self.w_steep_ret(array[1], array[2]) and self.w_rr(array[4], array[3], array[2]) < 0.618
                                and num11 < num3 and flag3 and self.w_impulse_rule_3(num12, num3, num7)):
                            num14 = array[2] - array2[2]
                            num15 = self.w_rt_over(array2[2], array2[1], array[2])
                            if 0 < num15 and num15 <= num14 / 2.0:
                                elwave.AddSP("※ A Terminal pattern may have concluded with m2", 32)
        return self._tail(elwave, array, array2, num, num2)

    # ==================================================================
    # Numeric / geometry helpers (direct translations)
    # ==================================================================

    def w_adj_ws(self, m1_s, m1_e, k):
        if k == 0:
            raise ValueError("The number of waves cannot be 0")
        num = 0
        num2 = abs(k)
        list_ = []
        if k > 0:
            list_.append(m1_s)
            list_.append(m1_e)
            num = 2
            candidate = num == num2
            s = m1_s
            num3 = m1_e
            num4, candidate = self.w_m2e(s, num3, candidate)
            while True:
                if not (num4 < len(self._plist) and num4 > num3):
                    break
                old = num
                num += 1
                if not (old <= num2):
                    break
                if candidate:
                    break
                list_.append(num4)
                s = num3
                num3 = num4
                candidate = num == num2
                num4, candidate = self.w_m2e(s, num3, candidate)
        else:
            s = m1_s
            num3 = m1_e
            num5 = self.w_m0s(s, num3)
            while True:
                if not (num5 >= 0 and num5 < s):
                    break
                old = num
                num += 1
                if not (old <= num2):
                    break
                list_.append(num5)
                num3 = s
                s = num5
                num5 = self.w_m0s(s, num3)
        return list_

    def w_dir_move_end(self, s1, e1):
        result = True
        candidate = False
        e2, candidate = self.w_m2e(s1, e1, candidate)
        num = self.w_m0s(e1, e2)
        e2, candidate = self.w_m2e(num, e1, candidate)
        num2 = self.w_rr(e2, e1, num)
        if num2 < 0.618:
            if num < s1 and self.w_rr(e2, e1, s1) > 1.0:
                m2e, candidate = self.w_m2e(e1, e2, candidate)
                if self.w_rr(m2e, e2, e1) < 1.618:
                    return result
            result = False
        elif num2 < 1.0:
            m2e, candidate = self.w_m2e(e1, e2, candidate)
            if self.w_rr(m2e, e2, e1) > 2.618:
                result = False
        return result

    def w_l_v(self, a, b):
        return abs(self._plist[a] - self._plist[b])

    def w_rr(self, a, b, c):
        return _div(self.w_l_v(a, b), self.w_l_v(b, c))

    def w_almost_same(self, d1, d2):
        result = False
        num = max(d1, d2)
        num2 = min(d1, d2)
        if num2 != 0.0:
            result = num / num2 < 1.1
        return result

    def w_Fibonacci_61_8(self, d1, d2):
        num = max(d1, d2)
        num2 = min(d1, d2)
        if 0.58 < _div(num2, num):
            return _div(num2, num) < 0.66
        return False

    def w_almost_Fibonacci_61_8(self, d1, d2):
        num = max(d1, d2)
        num2 = min(d1, d2)
        if 0.5562 < _div(num2, num):
            return _div(num2, num) < 0.6798
        return False

    def w_byond(self, s, e, e2):
        return (self._plist[e] - self._plist[s]) * (self._plist[e2] - self._plist[e]) > 0.0

    def w_mw_cnt(self, a, b):
        if a not in self._mw_list:
            return 0
        num = 0
        liE = self._mw_list[a].liE
        while liE <= b:
            num += 1
            if liE == b:
                break
            if liE in self._mw_list:
                liE = self._mw_list[liE].liE
                continue
            num += 1
            break
        return num

    def w_steep_ret(self, s, e):
        result = False
        num = e - s
        num2 = e
        while num != 0 and num2 < len(self._plist) - 1:
            num -= 1
            num2 += 1
            if _div(self._plist[num2] - self._plist[e], self._plist[e] - self._plist[s]) < -1.0:
                result = True
                num = 0
        return result

    def w_t_similar(self, s1, e1, s2, e2):
        num = e1 - s1
        num2 = e2 - s2
        if not self.w_almost_same(num, num2):
            return self.w_Fibonacci_61_8(num, num2)
        return True

    def w_p_similar(self, s1, e1, s2, e2):
        d = self.w_l_v(s1, e1)
        d2 = self.w_l_v(s2, e2)
        if not self.w_almost_same(d, d2):
            return self.w_Fibonacci_61_8(d, d2)
        return True

    def w_similar(self, s1, e1, s2, e2):
        if self.w_t_similar(s1, e1, s2, e2):
            return self.w_p_similar(s1, e1, s2, e2)
        return False

    def w_compare_slope(self, s1, e1, s2, e2):
        return _div((self._plist[e1] - self._plist[s1]) * (e2 - s2),
                    (self._plist[e2] - self._plist[s2]) * (e1 - s1))

    def w_steeper(self, s1, e1, s2, e2):
        if self.w_compare_slope(s1, e1, s2, e2) > 1.0:
            return self.w_l_v(s1, e1) > self.w_l_v(s2, e2)
        return False

    def w_overlap(self, s1, e1, s2, e2):
        num = max(self._plist[s1], self._plist[e1])
        num2 = min(self._plist[s1], self._plist[e1])
        num3 = max(self._plist[s2], self._plist[e2])
        num4 = min(self._plist[s2], self._plist[e2])
        if num3 >= num2:
            return num4 <= num
        return False

    def w_impulse_rule_3(self, len1, len3, len5):
        return len3 >= min(len1, len5)

    def w_impulse_rule_l(self, len1, len3, len5):
        result = True
        num = max(len1, max(len3, len5))
        if len1 == num:
            if len1 < 1.618 * max(len3, len5):
                result = False
        elif len3 == num:
            if len3 < 1.618 * max(len1, len5):
                result = False
        elif len5 < 1.618 * max(len1, len3):
            result = False
        return result

    def w_impulse_rule_r(self, len1, len3, len5, len6):
        return len6 >= 0.618 * len5

    def w_impulse_rule(self, len1, len3, len5, len6):
        if self.w_impulse_rule_3(len1, len3, len5) and self.w_impulse_rule_l(len1, len3, len5):
            return self.w_impulse_rule_r(len1, len3, len5, len6)
        return False

    def w_rt_time(self, a, b, c):
        if self.w_rr(c, b, a) <= 1.0:
            return 0
        i = b + 1
        while i != c and not (self.w_rr(i, b, a) > 1.0):
            i += 1
        return i - b

    def w_total_ret(self, s, e):
        result = 0
        num = e
        flag = False
        while num < len(self._plist) - 1 and not flag:
            num += 1
            flag = _div(self._plist[num] - self._plist[e], self._plist[e] - self._plist[s]) < -1.0
        if flag:
            result = num - e
        return result

    def w_tr_end(self, mn2e, m0e, m1e, m2e):
        flag = False
        num = _div(self._plist[m0e] - self._plist[mn2e], m0e - mn2e)
        num2 = m1e + 1
        num3 = m1e - m0e
        num4 = m2e - m1e
        while num3 > 0 and num4 > 0 and not flag:
            num3 -= 1
            num4 -= 1
            flag = _div(self._plist[num2] - self._plist[mn2e], num2 - mn2e) < num
            num2 += 1
        return flag

    def w_rt_over(self, s, e, s2):
        flag = False
        num = s2
        result = 0
        while num < len(self._plist) - 1 and not flag:
            num += 1
            flag = _div(self._plist[num] - self._plist[e], self._plist[e] - self._plist[s]) < -1.0
        if flag:
            result = num - s2
        return result

    def w_r_3w(self, s1, e1, e2):
        result = 0.0
        if self.w_mw_cnt(e1, e2) > 3:
            num = 0.0
            liE = self._mw_list[e1].liE
            for _ in range(3):
                _ = self._plist[liE - 1]
                num2 = self._plist[liE]
                _ = self._plist[liE + 1]
                num = min(num, _div(num2 - self._plist[e1], self._plist[e1] - self._plist[s1]))
                liE = self._mw_list[liE].liE
            result = num * -1.0
        return result

    def w_sim_nw(self, s, e):
        result = True
        candidate = False
        num, candidate = self.w_m2e(s, e, candidate)
        num2 = e - s
        num3 = num - e
        num4 = self.w_l_v(s, e)
        num5 = self.w_l_v(e, num)
        if 3.0 * num5 < num4 and 3 * num3 < num2:
            num6, candidate = self.w_m2e(e, num, candidate)
            result = (self.w_mw_cnt(num, num6) != 1
                      or not (self.w_l_v(num, num6) > 3.0 * num5)
                      or num6 - num <= 3 * num3)
        return result
